"""Gradio dashboard — polls state_log.jsonl and live ClickHouse telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

STATE_LOG = Path(__file__).parent / "state_log.jsonl"


# ---------------------------------------------------------------------------
# Local JSONL helpers
# ---------------------------------------------------------------------------

def _read_events() -> List[dict]:
    if not STATE_LOG.exists():
        return []
    events = []
    with open(STATE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _build_table(events: List[dict]) -> List[List[str]]:
    return [
        [
            e.get("timestamp", "")[:19],
            str(e.get("step_count", "")),
            e.get("event", ""),
            str(e.get("last_exit_code", "")),
            str(e.get("rollback_count", "")),
            e.get("current_thought", "")[:120],
        ]
        for e in events
    ]


def _latest_thought(events: List[dict]) -> str:
    for e in reversed(events):
        if e.get("current_thought"):
            return e["current_thought"]
    return "No events yet..."


def _status_summary(events: List[dict]) -> str:
    if not events:
        return "Waiting for agent to start..."
    last = events[-1]
    event_type = last.get("event", "")
    step = last.get("step_count", 0)
    rollbacks = last.get("rollback_count", 0)
    if event_type == "success":
        return f"SUCCESS — step {step} | {rollbacks} rollback(s)"
    if event_type == "rollback_executed":
        return f"ROLLBACK #{rollbacks} triggered at step {step}"
    if event_type == "max_steps":
        return f"STOPPED — max steps ({step}) | {rollbacks} rollback(s)"
    return f"Running... step {step} | rollbacks={rollbacks}"


def _rollback_events(events: List[dict]) -> List[List[str]]:
    rows = [
        [
            e.get("timestamp", "")[:19],
            str(e.get("step_count", "")),
            e.get("event", ""),
            e.get("current_thought", "")[:100],
        ]
        for e in events
        if e.get("event") in ("rollback_triggered", "rollback_executed")
    ]
    return rows if rows else [["—", "—", "—", "No rollbacks yet"]]


# ---------------------------------------------------------------------------
# ClickHouse live telemetry
# ---------------------------------------------------------------------------

def _read_clickhouse() -> List[List[str]]:
    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        return [["ClickHouse not configured", "", "", "", "", "", ""]]
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=host,
            username=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            secure=True,
        )
        rows = client.query(
            "SELECT timestamp, session_id, step_count, current_thought, "
            "exit_code, decision, rollback_count "
            "FROM gitagent_telemetry "
            "ORDER BY timestamp DESC LIMIT 50"
        ).result_rows
        return [
            [str(r[0])[:19], r[1], str(r[2]), r[3][:80], str(r[4]), r[5], str(r[6])]
            for r in rows
        ] or [["No rows yet", "", "", "", "", "", ""]]
    except Exception as exc:
        return [[f"Error: {exc}", "", "", "", "", "", ""]]


# ---------------------------------------------------------------------------
# Combined refresh
# ---------------------------------------------------------------------------

def refresh():
    events = _read_events()
    return (
        _build_table(events),
        _latest_thought(events),
        _status_summary(events),
        _rollback_events(events),
        _read_clickhouse(),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

STEP_HEADERS = ["Timestamp", "Step", "Event", "Exit Code", "Rollbacks", "Thought"]
ROLLBACK_HEADERS = ["Timestamp", "Step", "Event", "Thought"]
CH_HEADERS = ["Timestamp", "Session", "Step", "Thought (80c)", "Exit", "Decision", "Rollbacks"]

with gr.Blocks(title="GitAgent Dashboard") as demo:
    gr.Markdown("# GitAgent Live Dashboard")
    gr.Markdown(
        "Real-time view of the LangGraph agent loop. "
        "**ClickHouse Telemetry** tab shows the live database rows used for loop detection."
    )

    with gr.Row():
        status_box = gr.Textbox(label="Status", interactive=False)
        refresh_btn = gr.Button("Refresh", variant="primary")

    with gr.Tab("Step Trace"):
        step_table = gr.Dataframe(headers=STEP_HEADERS, datatype=["str"] * 6,
                                  interactive=False, wrap=True)

    with gr.Tab("Latest Thought"):
        thought_box = gr.Textbox(label="Agent's Last Thought", lines=12, interactive=False)

    with gr.Tab("Rollback Events"):
        rollback_table = gr.Dataframe(headers=ROLLBACK_HEADERS, datatype=["str"] * 4,
                                      interactive=False, wrap=True)

    with gr.Tab("ClickHouse Telemetry"):
        gr.Markdown("Live rows from `gitagent_telemetry` — this is what the loop detector queries.")
        ch_table = gr.Dataframe(headers=CH_HEADERS, datatype=["str"] * 7,
                                interactive=False, wrap=True)

    outputs = [step_table, thought_box, status_box, rollback_table, ch_table]

    refresh_btn.click(fn=refresh, outputs=outputs)

    timer = gr.Timer(3)
    timer.tick(fn=refresh, outputs=outputs)

    demo.load(fn=refresh, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Monochrome())
