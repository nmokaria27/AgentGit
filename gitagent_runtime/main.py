"""GitAgent — LangGraph workflow with real ClickHouse telemetry, Composio file/shell tools,
and MLflow experiment tracking."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, TypedDict

import anthropic
import clickhouse_connect
import mlflow
from composio_langgraph import ComposioToolSet, Action
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from evaluator import evaluate_trajectory_anomaly
from git_utils import init_workspace, rollback_to_baseline

load_dotenv()

SESSION_ID = str(uuid.uuid4())[:8]
STATE_LOG = str(Path(__file__).parent / "state_log.jsonl")


# ---------------------------------------------------------------------------
# Composio HostShell patch — fixes SHELLTOOL on macOS/Python 3.13
#
# Root cause: _has_command_exited checks full cmd strings against ps -e
# (which only shows short process names), returns True immediately before
# any output is buffered, causing empty-buffer returns.
#
# Fix: sentinel-based exec — append "echo SENTINEL:$?" after the command,
# read until that line appears, parse exit code from it. No ps polling.
# ---------------------------------------------------------------------------

def _patch_composio_shell() -> None:
    import os as _os
    import select as _sel
    import time as _time
    import typing as _t

    from composio.tools.env.host.shell import HostShell
    from composio.tools.env.constants import EXIT_CODE as _EXIT, STDOUT as _OUT, STDERR as _ERR

    def _sentinel_exec(self, cmd: str) -> dict:
        ts = int(_time.time() * 1000)
        sentinel = f"__GITAGENT_DONE_{ts}__"
        # Run command, capture its exit code, echo sentinel
        full = f"{cmd}\n__ec__=$?\nprintf '\\n{sentinel}:%s\\n' $__ec__\n"
        self._write(full)

        stdout_fd = _t.cast(_t.IO[str], self._process.stdout).fileno()
        stderr_fd = _t.cast(_t.IO[str], self._process.stderr).fileno()
        out_buf = b""
        err_buf = b""
        deadline = _time.time() + 120.0

        while _time.time() < deadline:
            rlist, _, _ = _sel.select([stdout_fd, stderr_fd], [], [], 0.3)
            if rlist:
                for fd in rlist:
                    data = _os.read(fd, 4096)
                    if data:
                        if fd == stdout_fd:
                            out_buf += data
                        else:
                            err_buf += data
            if sentinel.encode() in out_buf:
                break
        else:
            raise TimeoutError(f"Sentinel never appeared. stdout so far: {out_buf[:200]}")

        out_str = out_buf.decode(errors="replace")
        exit_code = 1
        clean_lines = []
        for line in out_str.splitlines():
            if sentinel in line:
                try:
                    exit_code = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    exit_code = 1
            else:
                clean_lines.append(line)

        return {
            _OUT: "\n".join(clean_lines),
            _ERR: err_buf.decode(errors="replace"),
            _EXIT: exit_code,
        }

    HostShell.exec = _sentinel_exec
    print("[Patch] Composio HostShell.exec patched with sentinel strategy")


_patch_composio_shell()

# Composio toolset — used for all file read/write and pytest execution
_composio = ComposioToolSet(api_key=os.environ.get("COMPOSIO_API_KEY", ""))

# Create a persistent shell at startup for pytest execution
_SHELL_ID: str = ""


def _ensure_shell() -> str:
    global _SHELL_ID
    if not _SHELL_ID:
        try:
            result = _composio.execute_action(
                action=Action.SHELLTOOL_CREATE_SHELL,
                params={},
                entity_id="default",
            )
            if result.get("successfull"):
                _SHELL_ID = result["data"]["shell_id"]
                print(f"[Composio] Persistent shell created: {_SHELL_ID}")
            else:
                print(f"[Composio] CreateShell failed: {result.get('error')}")
        except Exception as exc:
            print(f"[Composio] Shell init error: {exc}")
    return _SHELL_ID


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task: str
    step_count: int
    current_thought: str
    previous_thought: str
    last_exit_code: int
    exit_code_history: List[int]  # rolling list of exit codes for stagnation detection
    workspace_path: str
    baseline_sha: str       # exact commit SHA captured at init — rollback always targets this
    messages: List[Dict[str, Any]]
    status: str
    rollback_count: int


# ---------------------------------------------------------------------------
# Composio file helpers
# ---------------------------------------------------------------------------

def _composio_read_file(path: str) -> str:
    """Read a file via Composio FILETOOL_OPEN_FILE."""
    result = _composio.execute_action(
        action=Action.FILETOOL_OPEN_FILE,
        params={"file_path": path},
        entity_id="default",
    )
    if result.get("successfull"):
        lines = result["data"].get("lines", {})
        return "".join(lines.values()) if isinstance(lines, dict) else str(lines)
    raise RuntimeError(f"[Composio] FILETOOL_OPEN_FILE failed: {result.get('error')}")


def _composio_write_file(path: str, content: str) -> None:
    """Write a file via Composio FILETOOL_WRITE."""
    result = _composio.execute_action(
        action=Action.FILETOOL_WRITE,
        params={"file_path": path, "text": content},
        entity_id="default",
    )
    if not result.get("successfull"):
        raise RuntimeError(f"[Composio] FILETOOL_WRITE failed: {result.get('error')}")


# ---------------------------------------------------------------------------
# ClickHouse helpers
# ---------------------------------------------------------------------------

def _get_ch_client():
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        secure=True,
    )


def _log_clickhouse(state: AgentState, decision: str) -> None:
    """Write telemetry row to ClickHouse Cloud."""
    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        print("  [ClickHouse] CLICKHOUSE_HOST not set — skipping.")
        return
    try:
        client = _get_ch_client()
        client.insert(
            "gitagent_telemetry",
            [[
                datetime.now(timezone.utc),
                SESSION_ID,
                state.get("step_count", 0),
                state.get("current_thought", "")[:2000],
                state.get("previous_thought", "")[:2000],
                state.get("last_exit_code", -1),
                decision,
                state.get("rollback_count", 0),
            ]],
            column_names=[
                "timestamp", "session_id", "step_count",
                "current_thought", "previous_thought",
                "exit_code", "decision", "rollback_count",
            ],
        )
        print(f"  [ClickHouse] Row written — decision={decision}")
    except Exception as exc:
        print(f"  [ClickHouse] Write error: {exc}")


def _clickhouse_eval_decision(state: AgentState) -> str:
    """
    Query the last 2 rows for this session from ClickHouse and use them
    as the source of truth for loop/stagnation/give-up detection.
    Falls back to in-memory if ClickHouse has fewer than 2 rows yet.
    """
    history = state.get("exit_code_history", [])
    host = os.environ.get("CLICKHOUSE_HOST")
    if not host:
        return evaluate_trajectory_anomaly(
            state.get("current_thought", ""),
            state.get("previous_thought", ""),
            state["last_exit_code"],
            history,
        )
    try:
        client = _get_ch_client()
        rows = client.query(
            "SELECT current_thought, exit_code FROM gitagent_telemetry "
            "WHERE session_id = {sid:String} "
            "ORDER BY timestamp DESC LIMIT 2",
            parameters={"sid": SESSION_ID},
        ).result_rows

        if len(rows) >= 2:
            ch_current_thought, ch_exit_code = rows[0]
            ch_previous_thought, _ = rows[1]
            print(f"  [ClickHouse] Queried last 2 rows for session {SESSION_ID}")
            return evaluate_trajectory_anomaly(ch_current_thought, ch_previous_thought, ch_exit_code, history)
        else:
            print(f"  [ClickHouse] Only {len(rows)} row(s) — using in-memory state for eval")
            return evaluate_trajectory_anomaly(
                state.get("current_thought", ""),
                state.get("previous_thought", ""),
                state["last_exit_code"],
                history,
            )
    except Exception as exc:
        print(f"  [ClickHouse] Query error: {exc} — falling back to in-memory")
        return evaluate_trajectory_anomaly(
            state.get("current_thought", ""),
            state.get("previous_thought", ""),
            state["last_exit_code"],
            history,
        )


# ---------------------------------------------------------------------------
# State logger
# ---------------------------------------------------------------------------

def _log_state(state: AgentState, event: str) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": SESSION_ID,
        "event": event,
        "step_count": state.get("step_count", 0),
        "current_thought": state.get("current_thought", ""),
        "last_exit_code": state.get("last_exit_code", -1),
        "rollback_count": state.get("rollback_count", 0),
    }
    with open(STATE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Pytest runner — uses Composio SHELLTOOL_EXEC_COMMAND via patched sentinel exec
# ---------------------------------------------------------------------------

def _run_pytest(workspace: str) -> tuple[str, int]:
    """Run pytest via Composio SHELLTOOL_EXEC_COMMAND (patched sentinel strategy)."""
    shell_id = _ensure_shell()
    if shell_id:
        try:
            # cd first (fast — NOWAIT in original code, fine with our patch too)
            _composio.execute_action(
                action=Action.SHELLTOOL_EXEC_COMMAND,
                params={"cmd": f"cd {workspace}", "shell_id": shell_id},
                entity_id="default",
            )
            # Run pytest
            result = _composio.execute_action(
                action=Action.SHELLTOOL_EXEC_COMMAND,
                params={
                    "cmd": "python -m pytest test_math.py -v --tb=short -x",
                    "shell_id": shell_id,
                },
                entity_id="default",
            )
            if result.get("successfull"):
                data = result.get("data", {})
                output = data.get("stdout", "") + "\n" + data.get("stderr", "")
                exit_code = int(data.get("exit_code", 1))
                print(f"  [Composio] SHELLTOOL pytest exit_code={exit_code}")
                return output, exit_code
            else:
                print(f"  [Composio] SHELLTOOL failed: {result.get('error')} — fallback")
        except Exception as exc:
            print(f"  [Composio] SHELLTOOL error: {exc} — fallback to subprocess")

    # Fallback: plain subprocess
    proc = subprocess.run(
        ["python", "-m", "pytest", "test_math.py", "-v", "--tb=short", "-x"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(f"  [subprocess fallback] pytest exit_code={proc.returncode}")
    return proc.stdout + proc.stderr, proc.returncode


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_think_and_act(state: AgentState) -> Dict[str, Any]:
    workspace = state.get("workspace_path", "./mock_project")
    step = state.get("step_count", 1)
    src_path = os.path.join(workspace, "utils", "math_processor.py")

    print(f"\n[Step {step}] agent_think_and_act")

    # Read current source via Composio FILETOOL
    try:
        current_code = _composio_read_file(src_path)
        print("  [Composio] FILETOOL_OPEN_FILE — read math_processor.py")
    except Exception as exc:
        print(f"  [Composio] Read failed ({exc}), falling back to open()")
        with open(src_path) as f:
            current_code = f.read()

    # Run pytest via Composio SHELLTOOL
    pytest_output, exit_code = _run_pytest(workspace)
    print(f"  pytest exit_code={exit_code}")

    if exit_code == 0:
        thought = "All tests pass. Task complete."
        return {
            "step_count": step + 1,
            "current_thought": thought,
            "previous_thought": state.get("current_thought", ""),
            "last_exit_code": 0,
            "messages": state.get("messages", []),
            "status": "",
        }

    # Build messages for Claude
    messages: List[Dict[str, Any]] = list(state.get("messages", []))
    system_prompt = (
        "You are a Python debugging agent. Fix utils/math_processor.py so ALL pytest tests pass.\n"
        "Rules:\n"
        "- Respond with a short analysis (1-2 sentences) then the COMPLETE fixed file in a ```python block.\n"
        "- Do NOT repeat an approach you already tried.\n"
        "- If you are stuck, say GIVE_UP.\n"
    )
    if state.get("status"):
        system_prompt = state["status"] + "\n\n" + system_prompt

    user_msg = (
        f"Current utils/math_processor.py:\n```python\n{current_code}\n```\n\n"
        f"Pytest output (stopped at first failure):\n```\n{pytest_output}\n```\n\n"
        "Fix the code so all tests pass."
    )
    messages.append({"role": "user", "content": user_msg})

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    thought = response.content[0].text
    messages.append({"role": "assistant", "content": thought})
    print(f"  Agent thought: {thought[:140]}...")

    # Apply fix via Composio FILETOOL_WRITE
    code_match = re.search(r"```python\n(.*?)```", thought, re.DOTALL)
    if code_match:
        fixed = code_match.group(1).strip()
        try:
            _composio_write_file(src_path, fixed + "\n")
            print("  [Composio] FILETOOL_WRITE — wrote fix to math_processor.py")
        except Exception as exc:
            print(f"  [Composio] Write failed ({exc}), falling back to open()")
            with open(src_path, "w") as f:
                f.write(fixed + "\n")

        # Re-run to get exit code after fix
        _, exit_code = _run_pytest(workspace)
        print(f"  After fix: exit_code={exit_code}")

    new_state: Dict[str, Any] = {
        "step_count": step + 1,
        "current_thought": thought[:600],
        "previous_thought": state.get("current_thought", ""),
        "last_exit_code": exit_code,
        "exit_code_history": state.get("exit_code_history", []) + [exit_code],
        "messages": messages,
        "status": "",
    }
    _log_state({**state, **new_state}, "step")
    return new_state


def rollback_executor(state: AgentState) -> Dict[str, Any]:
    sha = state.get("baseline_sha", "HEAD")
    print(f"\n[ROLLBACK] Resetting workspace to baseline {sha[:8]}...")
    rollback_to_baseline(state["workspace_path"], sha)

    new_state: Dict[str, Any] = {
        "step_count": state.get("step_count", 1),
        "current_thought": "SYSTEM OVERRIDE: Workspace reset. Loop detected — prior approach aborted.",
        "previous_thought": "",
        "last_exit_code": 1,
        "messages": [],
        "exit_code_history": [],
        "baseline_sha": state.get("baseline_sha", "HEAD"),
        "rollback_count": state.get("rollback_count", 0) + 1,
        "status": (
            "CRITICAL WARNING: A repetitive loop was detected in your previous attempts. "
            "The workspace has been reset to baseline. "
            "ALL three functions (compute_percentage, compute_ratio, compute_weight) have the same bug. "
            "Fix all of them in one response."
        ),
    }
    _log_state({**state, **new_state}, "rollback_executed")
    return new_state


# ---------------------------------------------------------------------------
# Conditional edge — ClickHouse is the source of truth
# ---------------------------------------------------------------------------

def clickhouse_eval(state: AgentState) -> str:
    step = state.get("step_count", 1) - 1
    print(f"\n[Eval] step={step} exit_code={state['last_exit_code']}")

    if state["last_exit_code"] == 0:
        print("  -> DONE (tests pass)")
        _log_clickhouse(state, "DONE")
        _log_state(state, "success")
        mlflow.log_metric("final_step", step)
        mlflow.log_metric("rollback_count", state.get("rollback_count", 0))
        return "done"

    if state.get("step_count", 0) > 12:
        print("  -> DONE (max steps)")
        _log_clickhouse(state, "MAX_STEPS")
        _log_state(state, "max_steps")
        mlflow.log_metric("final_step", step)
        return "done"

    # Write to ClickHouse first, then query it back as the decision source
    _log_clickhouse(state, "EVALUATING")
    decision = _clickhouse_eval_decision(state)
    print(f"  Evaluator decision (from ClickHouse): {decision}")

    # Overwrite the placeholder row with the real decision
    _log_clickhouse(state, decision)

    mlflow.log_metric("step", step)
    mlflow.log_metric("exit_code", state["last_exit_code"])
    mlflow.log_metric("rollback_count", state.get("rollback_count", 0))

    if decision == "TRIGGER_ROLLBACK":
        print("  -> ROLLBACK (ClickHouse loop detected)")
        _log_state(state, "rollback_triggered")
        mlflow.log_metric("rollback_triggered_at_step", step)
        return "rollback"

    return "continue"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("agent_core", agent_think_and_act)
workflow.add_node("rollback_core", rollback_executor)

workflow.set_entry_point("agent_core")

workflow.add_conditional_edges(
    "agent_core",
    clickhouse_eval,
    {"rollback": "rollback_core", "continue": "agent_core", "done": END},
)
workflow.add_edge("rollback_core", "agent_core")

app = workflow.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

BROKEN_CODE = """\
def compute_percentage(part, total):
    \"\"\"Returns what percentage `part` is of `total`.\"\"\"\
    return (part / total) * 100


def compute_ratio(numerator, denominator):
    \"\"\"Returns decimal ratio of numerator to denominator.\"\"\"
    return numerator / denominator


def compute_weight(value, total_weight):
    \"\"\"Returns fractional weight of value within total_weight.\"\"\"
    return value / total_weight
"""

if __name__ == "__main__":
    workspace = os.path.abspath("./mock_project")

    # Always write the broken version before init so the git baseline captures it
    broken_path = os.path.join(workspace, "utils", "math_processor.py")
    os.makedirs(os.path.dirname(broken_path), exist_ok=True)
    with open(broken_path, "w") as f:
        f.write(BROKEN_CODE)

    print("Initializing git workspace...")
    baseline_sha = init_workspace(workspace)

    open(STATE_LOG, "w").close()

    # MLflow: start experiment run
    mlflow.set_experiment("gitagent")
    with mlflow.start_run(run_name=f"session-{SESSION_ID}") as run:
        mlflow.log_param("session_id", SESSION_ID)
        mlflow.log_param("workspace", workspace)
        mlflow.log_param("model", "claude-haiku-4-5-20251001")
        mlflow.log_param("loop_detector", "levenshtein>0.85")
        mlflow.log_param("baseline_sha", baseline_sha)

        initial: AgentState = {
            "task": "Fix all functions in math_processor.py so every test passes",
            "step_count": 1,
            "current_thought": "",
            "previous_thought": "",
            "last_exit_code": 1,
            "exit_code_history": [],
            "workspace_path": workspace,
            "baseline_sha": baseline_sha,
            "messages": [],
            "status": "",
            "rollback_count": 0,
        }

        print(f"\nStarting GitAgent (session={SESSION_ID})\n{'='*60}")
        print(f"[MLflow] Run ID: {run.info.run_id}")
        print(f"[MLflow] Track UI: mlflow ui --port 5001")
        for chunk in app.stream(initial, {"recursion_limit": 30}):
            pass

    print(f"\n{'='*60}\nDone. session={SESSION_ID} | Check state_log.jsonl, ClickHouse, and MLflow UI.")
