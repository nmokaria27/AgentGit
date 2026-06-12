import subprocess
import os


def init_workspace(path: str) -> str:
    """
    Initializes a clean git baseline inside the targeted sandbox path.
    Returns the baseline commit SHA so rollback always targets exactly
    this commit, regardless of any commits the agent makes later.
    """
    os.makedirs(path, exist_ok=True)
    try:
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init_baseline", "--allow-empty"],
            cwd=path,
            check=True,
            capture_output=True,
        )
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"Git baseline locked at {sha[:8]}.")
        return sha
    except subprocess.CalledProcessError as e:
        print(f"Failed to init git workspace: {e.stderr.decode()}")
        return "HEAD"


def rollback_to_baseline(path: str, baseline_sha: str = "HEAD") -> None:
    """
    Resets the workspace to the exact baseline commit captured at startup.
    Using the explicit SHA means rollback is safe even if the agent
    committed intermediate changes — it always goes back to the
    pre-agent state, not just the most recent commit.
    """
    try:
        subprocess.run(
            ["git", "reset", "--hard", baseline_sha],
            cwd=path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=path,
            check=True,
            capture_output=True,
        )
        print(f"Workspace restored to baseline {baseline_sha[:8]}.")
    except subprocess.CalledProcessError as e:
        print(f"Critical rollback failure: {e.stderr.decode()}")
