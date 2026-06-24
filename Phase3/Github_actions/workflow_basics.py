# phase3/04_github_actions/01_workflow_basics.py
# -------------------------------------------------------
# TOPIC: What happens inside a GitHub Actions workflow
#
# What you'll learn:
#   - How exit codes control CI pass/fail
#   - How secrets are read as environment variables
#   - What the regression script looks like end to end
#   - How the PR comment gets posted
#
# This file simulates what runs INSIDE the GitHub Actions VM.
# It is not a test of GitHub Actions itself —
# it is the script that GitHub Actions will run.
#
# Run locally: python 01_workflow_basics.py
# Run in CI:   triggered automatically by eval.yml
# -------------------------------------------------------

import asyncio
import os
import sys
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


# -------------------------------------------------------
# Data shapes
# -------------------------------------------------------

@dataclass
class RegressionDiff:
    test_id: str
    baseline_score: float
    current_score: float
    drop: float
    status: str          # REGRESSION / WARNING / IMPROVEMENT / STABLE
    reasoning: str = ""


@dataclass
class RegressionReport:
    run_id: str
    baseline_run_id: str
    total_cases: int
    regressions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    stable: list = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"""
Regression Report
{'─' * 40}
Baseline run  : {self.baseline_run_id}
Current run   : {self.run_id}
Total cases   : {self.total_cases}

Regressions   : {len(self.regressions)}
Warnings      : {len(self.warnings)}
Improvements  : {len(self.improvements)}
Stable        : {len(self.stable)}

CI Result     : {status}
{'─' * 40}"""

    def pr_comment_body(self) -> str:
        """Formatted markdown for the GitHub PR comment."""
        status = "❌ FAILED" if not self.passed else "✅ PASSED"
        body = f"""## EvalForge Regression Report

**Result:** {status}
| Metric | Count |
|---|---|
| Regressions | {len(self.regressions)} |
| Warnings | {len(self.warnings)} |
| Improvements | {len(self.improvements)} |
| Stable | {len(self.stable)} |

"""
        if self.regressions:
            body += "### ❌ Regressions (PR blocked)\n"
            for r in self.regressions:
                body += f"- **{r['test_id']}**: `{r['baseline_score']}` → `{r['current_score']}` "
                body += f"(drop `{r['drop']}`)\n"
                if r.get('reasoning'):
                    body += f"  > {r['reasoning']}\n"

        if self.warnings:
            body += "\n### ⚠️ Warnings\n"
            for w in self.warnings:
                body += f"- **{w['test_id']}**: `{w['baseline_score']}` → `{w['current_score']}` "
                body += f"(drop `{w['drop']}`)\n"

        if self.improvements:
            body += "\n### ✅ Improvements\n"
            for i in self.improvements:
                body += f"- **{i['test_id']}**: `{i['baseline_score']}` → `{i['current_score']}` ↑\n"

        return body


# -------------------------------------------------------
# DEMO 1 — Exit codes
# This is the most important concept:
# 0 = CI passes, 1 = CI fails
# -------------------------------------------------------

def demo_exit_codes():
    print("\n--- DEMO 1: Exit Codes Control CI ---")
    print("""
  sys.exit(0)  →  script succeeded  →  GitHub marks check GREEN  →  PR can merge
  sys.exit(1)  →  script failed     →  GitHub marks check RED    →  PR blocked

  This is the ONLY mechanism that controls CI pass/fail.
  It doesn't matter what you print — only the exit code matters.

  Same mechanism as pytest:
    All tests pass  →  pytest exits with 0
    Any test fails  →  pytest exits with 1
""")

    # Simulate a passing check
    report_pass = RegressionReport(
        run_id="run_002", baseline_run_id="run_001",
        total_cases=5, passed=True
    )
    code = 0 if report_pass.passed else 1
    print(f"  Passing report → exit code: {code} → CI: {'✅ GREEN' if code == 0 else '❌ RED'}")

    # Simulate a failing check
    report_fail = RegressionReport(
        run_id="run_003", baseline_run_id="run_001",
        total_cases=5, passed=False,
        regressions=[{"test_id": "tc_002", "baseline_score": 3.5,
                      "current_score": 2.0, "drop": 1.5,
                      "reasoning": "Confuses coroutines with threading"}]
    )
    code = 0 if report_fail.passed else 1
    print(f"  Failing report → exit code: {code} → CI: {'✅ GREEN' if code == 0 else '❌ RED'}")


# -------------------------------------------------------
# DEMO 2 — Reading secrets as environment variables
# In GitHub Actions, secrets are injected as env vars.
# Your script reads them exactly like any other env var.
# -------------------------------------------------------

def demo_secrets():
    print("\n--- DEMO 2: Secrets as Environment Variables ---")
    print("""
  In the workflow YAML:
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

  In your Python script:
    api_key = os.getenv("ANTHROPIC_API_KEY")

  They are identical — secrets just inject the value securely.
  Never print secrets, never hardcode them, never commit them.
""")

    # Show what the script sees
    api_key  = os.getenv("ANTHROPIC_API_KEY", "NOT_SET")
    db_url   = os.getenv("DATABASE_URL",       "NOT_SET")
    mock     = os.getenv("MOCK_MODE",          "true")
    pr_num   = os.getenv("PR_NUMBER",          "NOT_SET")
    gh_token = os.getenv("GITHUB_TOKEN",       "NOT_SET")

    print(f"  ANTHROPIC_API_KEY : {'SET ✅' if api_key != 'NOT_SET' else 'not set (expected in CI)'}")
    print(f"  DATABASE_URL      : {'SET ✅' if db_url  != 'NOT_SET' else 'not set (expected in CI)'}")
    print(f"  MOCK_MODE         : {mock}")
    print(f"  PR_NUMBER         : {pr_num}")
    print(f"  GITHUB_TOKEN      : {'SET ✅' if gh_token != 'NOT_SET' else 'not set (expected in CI)'}")


# -------------------------------------------------------
# DEMO 3 — Prompt hashing
# Store a fingerprint of your judge prompt with every run.
# If scores change, compare hashes to find the cause.
# -------------------------------------------------------

def demo_prompt_hashing():
    print("\n--- DEMO 3: Prompt Hashing ---")

    JUDGE_SYSTEM_PROMPT_V1 = "You are an eval judge. Score outputs 1.0-5.0."
    JUDGE_SYSTEM_PROMPT_V2 = "You are a strict eval judge. Score outputs 1.0-5.0."

    def hash_prompt(prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()[:8]

    hash_v1 = hash_prompt(JUDGE_SYSTEM_PROMPT_V1)
    hash_v2 = hash_prompt(JUDGE_SYSTEM_PROMPT_V2)

    print(f"  Prompt v1 hash: {hash_v1}")
    print(f"  Prompt v2 hash: {hash_v2}")
    print(f"  Hashes match  : {hash_v1 == hash_v2}")
    print("""
  When debugging a regression:

  Run 001: prompt_hash=a3f8c1d2, avg_score=4.1
  Run 002: prompt_hash=b7e2a9f4, avg_score=3.3  ← hash changed = prompt changed
  Run 003: prompt_hash=b7e2a9f4, avg_score=3.1  ← hash same = model or data changed

  Without hashing: "why did scores drop?" — no idea
  With hashing:    "prompt changed between run 001 and 002" — immediate answer
""")


# -------------------------------------------------------
# DEMO 4 — The full regression script
# This is what GitHub Actions actually runs.
# In real usage: connects to PostgreSQL, calls real Anthropic API.
# Here: uses mock data to show the complete flow.
# -------------------------------------------------------

BASELINE_SCORES = {
    "tc_001": 4.5,
    "tc_002": 3.5,
    "tc_003": 1.5,
    "tc_004": 4.0,
    "tc_005": 4.5,
}

CURRENT_SCORES = {
    "tc_001": 4.5,    # stable
    "tc_002": 2.0,    # regression — dropped 1.5
    "tc_003": 1.5,    # stable
    "tc_004": 3.8,    # stable (small drop, not threshold)
    "tc_005": 3.5,    # regression — dropped 1.0
}

REGRESSION_THRESHOLD = 1.0
WARNING_THRESHOLD    = 0.5


def run_regression_check(
    baseline: dict,
    current: dict,
) -> RegressionReport:
    """
    Compares current scores to baseline.
    In real usage this also runs the eval suite to get current scores.
    """
    regressions  = []
    warnings     = []
    improvements = []
    stable       = []

    for test_id, baseline_score in baseline.items():
        if test_id not in current:
            continue

        current_score = current[test_id]
        drop = round(baseline_score - current_score, 1)

        diff = {
            "test_id":        test_id,
            "baseline_score": baseline_score,
            "current_score":  current_score,
            "drop":           drop,
            "reasoning":      f"Mock reasoning for {test_id}"
        }

        if drop >= REGRESSION_THRESHOLD:
            regressions.append(diff)
        elif drop >= WARNING_THRESHOLD:
            warnings.append(diff)
        elif drop < 0:
            improvements.append(diff)
        else:
            stable.append(diff)

    return RegressionReport(
        run_id="run_007",
        baseline_run_id="run_001",
        total_cases=len(baseline),
        regressions=regressions,
        warnings=warnings,
        improvements=improvements,
        stable=stable,
        passed=len(regressions) == 0
    )


async def demo_full_script():
    print("\n--- DEMO 4: Full Regression Script ---")
    print("This is what GitHub Actions runs on every PR\n")

    # Step 1 — load baseline (from PostgreSQL in real usage)
    print("  Step 1: Loading baseline scores from database...")
    baseline = BASELINE_SCORES
    print(f"          Loaded {len(baseline)} baseline scores")

    # Step 2 — run current eval suite (calls real API in production)
    print("  Step 2: Running eval suite (mock)...")
    await asyncio.sleep(0.1)   # simulate eval run time
    current = CURRENT_SCORES
    print(f"          Scored {len(current)} test cases")

    # Step 3 — compare
    print("  Step 3: Comparing to baseline...\n")
    report = run_regression_check(baseline, current)

    # Step 4 — print report
    print(report.summary())

    # Step 5 — show what would be posted as PR comment
    print("\n  PR comment that would be posted:")
    print("  " + "─" * 40)
    for line in report.pr_comment_body().split("\n"):
        print(f"  {line}")
    print("  " + "─" * 40)

    # Step 6 — exit with correct code
    exit_code = 0 if report.passed else 1
    print(f"\n  sys.exit({exit_code}) → CI {'✅ PASSES' if exit_code == 0 else '❌ FAILS'}")
    print(f"  (not calling sys.exit here — demo mode)")

    return report


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("GITHUB ACTIONS — WHAT RUNS INSIDE THE VM")
    print("=" * 55)
    print(f"MOCK_MODE: {MOCK_MODE}")

    demo_exit_codes()
    demo_secrets()
    demo_prompt_hashing()
    await demo_full_script()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. sys.exit(1) = CI fails = PR blocked")
    print("  2. Secrets are just environment variables — read with os.getenv()")
    print("  3. Prompt hash tells you WHY scores changed")
    print("  4. The script is just Python — GitHub Actions just runs it")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())