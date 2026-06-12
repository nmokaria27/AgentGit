"""GitAgent — OpenUI Dashboard Server (FastAPI + SSE)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator, List

import anthropic
import clickhouse_connect
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

load_dotenv()

STATE_LOG = Path(__file__).parent / "state_log.jsonl"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="GitAgent Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Data helpers
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


def _read_clickhouse() -> List[dict]:
    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        return []
    try:
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
            {
                "timestamp": str(r[0])[:19],
                "session_id": r[1],
                "step_count": r[2],
                "thought": str(r[3])[:120],
                "exit_code": r[4],
                "decision": r[5],
                "rollback_count": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def get_state():
    events = _read_events()
    last = events[-1] if events else {}
    return {
        "events": events,
        "status": _derive_status(last),
        "latest_thought": next(
            (e["current_thought"] for e in reversed(events) if e.get("current_thought")),
            "Waiting for agent...",
        ),
        "rollbacks": [e for e in events if e.get("event") in ("rollback_triggered", "rollback_executed")],
    }


@app.get("/api/clickhouse")
def get_clickhouse():
    return {"rows": _read_clickhouse()}


@app.post("/api/reset")
def reset():
    open(STATE_LOG, "w").close()
    return {"ok": True}


@app.get("/api/summary")
async def get_summary():
    events = _read_events()
    if not events:
        return {"summary": "No events yet — run the agent first."}

    # Build compact per-step context for the model
    lines = []
    for e in events:
        thought = (e.get("current_thought") or "").replace("\n", " ")[:400]
        lines.append(
            f"step={e.get('step_count')}  event={e.get('event')}  "
            f"exit_code={e.get('last_exit_code')}  rollbacks={e.get('rollback_count')}\n"
            f"thought: {thought}"
        )
    context = "\n---\n".join(lines)

    prompt = (
        "You are summarising an AI agent's debugging run. "
        "Below is the full event log. Write a clear step-by-step account of what happened: "
        "what the agent observed each step, what fix it attempted, whether it worked, "
        "whether a rollback fired and why, and how the run ended. "
        "Use plain language, one short paragraph per step, no bullet points.\n\n"
        f"{context}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"summary": response.content[0].text}


def _derive_status(last: dict) -> dict:
    event = last.get("event", "")
    step = last.get("step_count", 0)
    rollbacks = last.get("rollback_count", 0)
    if not last:
        return {"label": "idle", "text": "Waiting for agent to start…", "color": "gray"}
    if event == "success":
        return {"label": "success", "text": f"✓ Done — step {step} | {rollbacks} rollback(s)", "color": "green"}
    if event == "rollback_executed":
        return {"label": "rollback", "text": f"↺ Rollback #{rollbacks} at step {step}", "color": "orange"}
    if event == "max_steps":
        return {"label": "stopped", "text": f"⚑ Max steps ({step}) | {rollbacks} rollback(s)", "color": "red"}
    return {"label": "running", "text": f"⟳ Running — step {step} | rollbacks: {rollbacks}", "color": "blue"}


# ---------------------------------------------------------------------------
# SSE stream — tails state_log.jsonl and pushes new events to the browser
# ---------------------------------------------------------------------------

async def _event_generator(request) -> AsyncGenerator[dict, None]:
    last_size = -1
    last_heartbeat = asyncio.get_event_loop().time()

    while True:
        if await request.is_disconnected():
            break

        now = asyncio.get_event_loop().time()

        try:
            size = STATE_LOG.stat().st_size if STATE_LOG.exists() else 0

            if size != last_size:
                last_size = size
                # Run blocking file I/O off the event loop
                events = await asyncio.to_thread(_read_events)
                last = events[-1] if events else {}
                payload = {
                    "events": events,
                    "status": _derive_status(last),
                    "latest_thought": next(
                        (e["current_thought"] for e in reversed(events) if e.get("current_thought")),
                        "Waiting for agent...",
                    ),
                    "rollbacks": [e for e in events if e.get("event") in ("rollback_triggered", "rollback_executed")],
                }
                last_heartbeat = now
                yield {"data": json.dumps(payload)}

            elif now - last_heartbeat > 5:
                # Keepalive so the browser doesn't close the SSE connection
                yield {"data": json.dumps({"heartbeat": True})}
                last_heartbeat = now

        except Exception:
            pass

        await asyncio.sleep(0.3)


@app.get("/api/stream")
async def stream(request: Request):
    return EventSourceResponse(_event_generator(request))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861, reload=False)
