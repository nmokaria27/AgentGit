import Levenshtein

# How many consecutive failed steps before we declare stagnation
STAGNATION_THRESHOLD = 4

# Rolling window of recent exit codes — maintained by the caller via exit_code_history
_exit_code_history: list[int] = []


def evaluate_trajectory_anomaly(
    current_thought: str,
    previous_thought: str,
    exit_code: int,
    exit_code_history: list[int] | None = None,
) -> str:
    """
    Detects agent anomalies and returns one of:
      CONTINUE         — nothing wrong, keep going
      TRIGGER_ROLLBACK — reset workspace and warn the agent

    Checks (in order):
      1. GIVE_UP signal  — agent explicitly said it's stuck
      2. Repetitive loop — consecutive thoughts are >85% similar
      3. Stagnation      — last N steps all failed with no progress
    """
    if exit_code == 0:
        return "CONTINUE"

    # 1. Agent gave up
    if "GIVE_UP" in current_thought.upper():
        return "TRIGGER_ROLLBACK"

    if not previous_thought:
        return "CONTINUE"

    # 2. Repetitive loop
    similarity_ratio = Levenshtein.ratio(current_thought, previous_thought)
    if similarity_ratio > 0.85:
        return "TRIGGER_ROLLBACK"

    # 3. Stagnation — all recent steps failed
    history = exit_code_history or []
    if len(history) >= STAGNATION_THRESHOLD:
        recent = history[-STAGNATION_THRESHOLD:]
        if all(ec != 0 for ec in recent):
            return "TRIGGER_ROLLBACK"

    return "CONTINUE"
