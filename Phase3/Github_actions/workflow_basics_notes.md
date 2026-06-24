# 01_workflow_basics.py — Full Walkthrough

## What this file is

This file simulates what happens **inside** the GitHub Actions VM when your regression script runs. It is not a test of GitHub Actions itself — it is the Python script that GitHub Actions will execute.

It has four demos that each teach one concept:

| Demo | What it teaches |
|---|---|
| Demo 1 | How exit codes control CI pass/fail |
| Demo 2 | How secrets become environment variables |
| Demo 3 | How prompt hashing detects prompt changes |
| Demo 4 | The complete regression pipeline end to end |

---

## Entry point

```python
if __name__ == "__main__":
    asyncio.run(main())
```

Python jumps here first. `asyncio.run(main())` starts the async event loop and calls `main()`. Everything happens inside `main()`.

---

## main() — runs four demos in order

```python
async def main():
    demo_exit_codes()
    demo_secrets()
    demo_prompt_hashing()
    await demo_full_script()
```

Four functions called one after the other. Each is completely independent — they do not share data. The first three are regular functions. The last one is `async` so it needs `await`.

---

## Demo 1 — demo_exit_codes()

### What it teaches

The entire CI pass/fail mechanism is one number — the exit code. `0` means success, anything else means failure.

### What it does

```python
report_pass = RegressionReport(
    run_id="run_002", baseline_run_id="run_001",
    total_cases=5, passed=True
)
code = 0 if report_pass.passed else 1
```

Creates a `RegressionReport` dataclass with `passed=True`. Then computes the exit code with a ternary:

```
passed=True  → code = 0  → CI green  → PR can merge
passed=False → code = 1  → CI red    → PR blocked
```

Does the same with a failing report:

```python
report_fail = RegressionReport(
    run_id="run_003", baseline_run_id="run_001",
    total_cases=5, passed=False,
    regressions=[{"test_id": "tc_002", ...}]
)
code = 0 if report_fail.passed else 1
```

**What's in memory:**
```
report_pass.passed = True   → code = 0
report_fail.passed = False  → code = 1
```

### Why sys.exit() is never actually called here

If the demo called `sys.exit(1)`, the entire script would terminate immediately and demos 2, 3, and 4 would never run. The demo only prints what the exit code would be — in the real `run_regression.py` it becomes `sys.exit(exit_code)` for real.

### The key insight

```
pytest:     all tests pass → exits 0 → CI green
            any test fails → exits 1 → CI red

EvalForge:  no regressions → exits 0 → CI green
            regressions    → exits 1 → CI red
```

Same mechanism. Same exit code convention. GitHub Actions reads it the same way.

---

## Demo 2 — demo_secrets()

### What it teaches

Secrets in GitHub are just environment variables. Your Python script reads them with `os.getenv()` — identically whether running locally or inside GitHub Actions.

### What it does

```python
api_key  = os.getenv("ANTHROPIC_API_KEY", "NOT_SET")
db_url   = os.getenv("DATABASE_URL",       "NOT_SET")
mock     = os.getenv("MOCK_MODE",          "true")
pr_num   = os.getenv("PR_NUMBER",          "NOT_SET")
gh_token = os.getenv("GITHUB_TOKEN",       "NOT_SET")
```

`os.getenv("KEY", "DEFAULT")` reads an environment variable. If the variable is not set, returns the default value instead of crashing.

### What's in memory — locally vs in CI

**Running locally (your machine):**
```
api_key  = "NOT_SET"     (unless you have it in .env)
db_url   = "NOT_SET"
mock     = "true"        (from your .env file)
pr_num   = "NOT_SET"
gh_token = "NOT_SET"
```

**Running inside GitHub Actions:**
```
api_key  = "sk-ant-..."      (injected from secrets)
db_url   = "postgresql://..."(injected from secrets)
mock     = "false"           (hardcoded in workflow YAML)
pr_num   = "42"              (injected from github.event)
gh_token = "ghs_abc..."      (auto-provided by GitHub)
```

Same code. Same `os.getenv()` calls. Different values depending on environment. This is the whole point — the script does not know or care whether it is local or in CI.

### Why the second argument matters

```python
os.getenv("ANTHROPIC_API_KEY")           # returns None if not set
os.getenv("ANTHROPIC_API_KEY", "NOT_SET") # returns "NOT_SET" if not set
```

Without a default, a missing variable returns `None`. Later when you try to use it as a string — `"Bearer " + api_key` — Python crashes with `TypeError: can only concatenate str (not "NoneType") to str`. The default prevents this.

---

## Demo 3 — demo_prompt_hashing()

### What it teaches

Storing a fingerprint (hash) of your judge system prompt with every eval run lets you diagnose why scores changed. If hashes differ between runs, the prompt changed. If hashes match, something else changed (model update, data change).

### What it does

```python
JUDGE_SYSTEM_PROMPT_V1 = "You are an eval judge. Score outputs 1.0-5.0."
JUDGE_SYSTEM_PROMPT_V2 = "You are a strict eval judge. Score outputs 1.0-5.0."
```

Two prompt strings. One word different — "strict" added.

```python
def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:8]
```

Breaking this down:
- `prompt.encode()` — converts the string to bytes. MD5 requires bytes, not a string.
- `hashlib.md5(...)` — creates an MD5 hash object — a mathematical fingerprint of the bytes.
- `.hexdigest()` — converts the hash to a readable hex string like `"a3f8c1d2b4e69f21"`.
- `[:8]` — takes just the first 8 characters. Short enough to store, unique enough to be useful.

```python
hash_v1 = hash_prompt(JUDGE_SYSTEM_PROMPT_V1)   # e.g. "a3f8c1d2"
hash_v2 = hash_prompt(JUDGE_SYSTEM_PROMPT_V2)   # e.g. "b7e2a9f4"
```

**What's in memory:**
```
hash_v1 = "a3f8c1d2"   (some 8-char hex string)
hash_v2 = "b7e2a9f4"   (completely different — one word changed)
```

Even a single character change produces a totally different hash. This is the key property — you cannot get the same hash from two different strings.

```python
print(f"  Hashes match: {hash_v1 == hash_v2}")
# → False
```

### How this helps you debug regressions

```
Run 001: prompt_hash=a3f8c1d2, avg_score=4.1
Run 002: prompt_hash=b7e2a9f4, avg_score=3.3  ← hash changed = prompt changed
Run 003: prompt_hash=b7e2a9f4, avg_score=3.1  ← hash same = something else changed
```

Without hashing: "why did scores drop between run 001 and 002?" — no idea.
With hashing: "the prompt changed" — immediate answer. You check your git diff for the prompt file and find the edit.

---

## Demo 4 — demo_full_script()

### What it teaches

The complete picture. Every concept from Phase 3 in one flow: load baseline → run eval suite → compare scores → build report → determine exit code.

### The data used

```python
BASELINE_SCORES = {
    "tc_001": 4.5,
    "tc_002": 3.5,
    "tc_003": 1.5,
    "tc_004": 4.0,
    "tc_005": 4.5,
}

CURRENT_SCORES = {
    "tc_001": 4.5,    # stable — no change
    "tc_002": 2.0,    # dropped 1.5 — regression
    "tc_003": 1.5,    # stable — no change
    "tc_004": 3.8,    # dropped 0.2 — too small to be a warning
    "tc_005": 3.5,    # dropped 1.0 — regression
}
```

Module-level constants — they exist before any function runs. In real usage:
- `BASELINE_SCORES` comes from a PostgreSQL query
- `CURRENT_SCORES` comes from running the full eval suite through the Anthropic API

Here they are hardcoded so you can see the flow without a database or API key.

---

### Step 1 — Load baseline

```python
baseline = BASELINE_SCORES
print(f"  Loaded {len(baseline)} baseline scores")
```

**What's in memory:**
```
baseline = {"tc_001": 4.5, "tc_002": 3.5, "tc_003": 1.5, "tc_004": 4.0, "tc_005": 4.5}
```

In real usage this is `await load_baseline("run_001", db)` — a PostgreSQL query that returns all scores from the last known good run.

---

### Step 2 — Run eval suite

```python
await asyncio.sleep(0.1)
current = CURRENT_SCORES
```

`asyncio.sleep(0.1)` simulates the time a real eval run takes — typically 5–30 seconds for 20 test cases running concurrently. In real usage, this is `asyncio.gather()` firing 20 API calls at once.

**What's in memory:**
```
current = {"tc_001": 4.5, "tc_002": 2.0, "tc_003": 1.5, "tc_004": 3.8, "tc_005": 3.5}
```

---

### Step 3 — Compare: run_regression_check()

```python
report = run_regression_check(baseline, current)
```

Inside `run_regression_check()`, a loop iterates every test case:

```python
for test_id, baseline_score in baseline.items():
    current_score = current[test_id]
    drop = round(baseline_score - current_score, 1)
```

Computes the drop for each:
```
tc_001: 4.5 - 4.5 = 0.0
tc_002: 3.5 - 2.0 = 1.5  ← big drop
tc_003: 1.5 - 1.5 = 0.0
tc_004: 4.0 - 3.8 = 0.2  ← tiny drop
tc_005: 4.5 - 3.5 = 1.0  ← drop
```

Then classifies each result:

```python
REGRESSION_THRESHOLD = 1.0
WARNING_THRESHOLD    = 0.5

if drop >= REGRESSION_THRESHOLD:     # >= 1.0
    regressions.append(diff)
elif drop >= WARNING_THRESHOLD:      # >= 0.5
    warnings.append(diff)
elif drop < 0:                       # score went UP
    improvements.append(diff)
else:                                # small or no change
    stable.append(diff)
```

Classification of each test case:
```
tc_001: drop 0.0 → stable        (0.0 < 0.5)
tc_002: drop 1.5 → REGRESSION    (1.5 >= 1.0)
tc_003: drop 0.0 → stable        (0.0 < 0.5)
tc_004: drop 0.2 → stable        (0.2 < 0.5)
tc_005: drop 1.0 → REGRESSION    (1.0 >= 1.0)
```

**What's in memory after the loop:**
```
regressions  = [tc_002 diff, tc_005 diff]
warnings     = []
improvements = []
stable       = [tc_001 diff, tc_003 diff, tc_004 diff]
```

Then builds the report:

```python
return RegressionReport(
    regressions=regressions,
    warnings=warnings,
    improvements=improvements,
    stable=stable,
    passed=len(regressions) == 0    # False — 2 found
)
```

`passed=False` because `len(regressions) == 2`, not 0.

---

### Step 4 — Print the summary

```python
print(report.summary())
```

`summary()` is a method on `RegressionReport`. It reads `self.regressions`, `self.warnings`, etc. and builds a formatted string. No logic here — just formatting and printing.

---

### Step 5 — Show the PR comment

```python
for line in report.pr_comment_body().split("\n"):
    print(f"  {line}")
```

`pr_comment_body()` builds a markdown string — formatted with headers, tables, and bullet points. This is what gets POSTed to the GitHub API and appears as a comment on the PR. Here it is just printed so you can see what it looks like.

---

### Step 6 — The exit code

```python
exit_code = 0 if report.passed else 1
print(f"\n  sys.exit({exit_code}) → CI {'✅ PASSES' if exit_code == 0 else '❌ FAILS'}")
print(f"  (not calling sys.exit here — demo mode)")
```

`report.passed = False` → `exit_code = 1`.

In the real `run_regression.py` this becomes `sys.exit(exit_code)` — not a print. That one line terminates the script with exit code 1, GitHub reads it, marks the check failed, and the PR is blocked.

---

## Complete memory picture at end of Demo 4

```
baseline    = {tc_001: 4.5, tc_002: 3.5, tc_003: 1.5, tc_004: 4.0, tc_005: 4.5}
current     = {tc_001: 4.5, tc_002: 2.0, tc_003: 1.5, tc_004: 3.8, tc_005: 3.5}

report      = RegressionReport(
                regressions  = [tc_002 diff, tc_005 diff]
                warnings     = []
                improvements = []
                stable       = [tc_001 diff, tc_003 diff, tc_004 diff]
                passed       = False
              )

exit_code   = 1
```

Two test cases regressed. `passed=False`. Exit code would be 1. CI would block the PR.

---

## The full execution flow

```
asyncio.run(main())
  │
  ├── demo_exit_codes()
  │     creates RegressionReport(passed=True)  → code = 0
  │     creates RegressionReport(passed=False) → code = 1
  │     prints both — never calls sys.exit()
  │
  ├── demo_secrets()
  │     reads 5 environment variables with os.getenv()
  │     prints SET or NOT_SET for each
  │
  ├── demo_prompt_hashing()
  │     hashes two prompts with hashlib.md5()
  │     prints both hashes — different even though one word changed
  │
  └── demo_full_script()
        baseline = BASELINE_SCORES (hardcoded dict)
        current  = CURRENT_SCORES  (hardcoded dict)
              ↓
        run_regression_check(baseline, current)
          loop: compute drop per test case
          classify: REGRESSION / WARNING / IMPROVEMENT / STABLE
          return RegressionReport(passed=False)
              ↓
        print report.summary()
        print report.pr_comment_body()
        print "sys.exit(1)" — but does not call it
```

---

## The one thing to remember

This file is the script that GitHub Actions runs. The VM, the secrets, the checkout — all of that is setup to get here. The actual work is just Python: read env vars, compare dicts, compute exit code. GitHub Actions does not do the thinking — it just runs your script and reads the exit code.