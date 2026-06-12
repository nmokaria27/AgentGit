"""Confirms all three evaluator triggers fire correctly."""
from evaluator import evaluate_trajectory_anomaly, STAGNATION_THRESHOLD
import Levenshtein as _lev

# ── Realistic loop thoughts: analysis differs, but the code block (all 3 functions)
# is nearly identical — exactly what Haiku generates each step. ──────────────────
LOOP_T1 = (
    "Need a zero-division guard in compute_percentage.\n"
    "```python\n"
    "def compute_percentage(part, total):\n"
    "    if total == 0: return 0.0\n"
    "    return (part / total) * 100\n\n"
    "def compute_ratio(numerator, denominator):\n"
    "    return numerator / denominator\n\n"
    "def compute_weight(value, total_weight):\n"
    "    return value / total_weight\n"
    "```"
)
LOOP_T2 = (
    "Need a zero-division guard in compute_ratio.\n"
    "```python\n"
    "def compute_percentage(part, total):\n"
    "    if total == 0: return 0.0\n"
    "    return (part / total) * 100\n\n"
    "def compute_ratio(numerator, denominator):\n"
    "    if denominator == 0: return 0.0\n"
    "    return numerator / denominator\n\n"
    "def compute_weight(value, total_weight):\n"
    "    return value / total_weight\n"
    "```"
)

DIFF_A = "Syntax error: docstring and return on the same line in compute_percentage."
DIFF_B = "Fixed indentation. Now the zero-division error surfaces in compute_ratio."

passed = 0

# ── 0. Sanity-check the similarity so the threshold is transparent ────────────
sim = round(_lev.ratio(LOOP_T1, LOOP_T2), 3)
print(f"   Loop thought similarity: {sim}  (threshold > 0.85)")
assert sim > 0.85, f"Test setup error: similarity {sim} not above threshold"

# ── 1. Loop (Levenshtein > 0.85) ─────────────────────────────────────────────
result = evaluate_trajectory_anomaly(LOOP_T2, LOOP_T1, exit_code=1)
assert result == "TRIGGER_ROLLBACK", f"Loop: got {result}"
print(f"✓  Loop detection           → {result}")
passed += 1

# ── 2. GIVE_UP ────────────────────────────────────────────────────────────────
for phrase in ["GIVE_UP", "give_up", "I must GIVE_UP — cannot proceed"]:
    result = evaluate_trajectory_anomaly(phrase, DIFF_A, exit_code=1)
    assert result == "TRIGGER_ROLLBACK", f"GIVE_UP '{phrase}': got {result}"
print(f"✓  GIVE_UP detection        → TRIGGER_ROLLBACK  (3 variants)")
passed += 1

# ── 3. Stagnation (N consecutive non-zero exit codes) ────────────────────────
stale = [1] * STAGNATION_THRESHOLD
result = evaluate_trajectory_anomaly(DIFF_B, DIFF_A, exit_code=1,
                                     exit_code_history=stale)
assert result == "TRIGGER_ROLLBACK", f"Stagnation: got {result}"
print(f"✓  Stagnation detection     → {result}  ({STAGNATION_THRESHOLD} consecutive failures)")
passed += 1

# ── 4. No false positives ─────────────────────────────────────────────────────
result = evaluate_trajectory_anomaly(DIFF_B, DIFF_A, exit_code=1,
                                     exit_code_history=[1, 0, 1])
assert result == "CONTINUE", f"False positive on mixed history: got {result}"
print(f"✓  No false positive        → {result}  (history has a 0)")
passed += 1

result = evaluate_trajectory_anomaly(LOOP_T1, LOOP_T1, exit_code=0)
assert result == "CONTINUE", f"exit_code=0 should always CONTINUE: got {result}"
print(f"✓  exit_code=0 always safe  → {result}")
passed += 1

print(f"\n{'='*50}")
print(f"  All {passed}/5 evaluator checks passed.")
print(f"{'='*50}")
