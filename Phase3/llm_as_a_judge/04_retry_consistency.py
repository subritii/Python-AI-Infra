# TOPIC: Retry logic and judge consistency
#
# What you'll learn:
#   - Why retry logic is essential for the judge
#   - How to give the judge error feedback on retry
#   - Why temperature=0.0 matters for consistency
#   - How to measure judge consistency yourself
#
# -------------------------------------------------------

import json
import asyncio
import os
from pydantic import BaseModel, field_validator, ValidationError
from typing import List
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------
# JudgeResponse model
# -------------------------------------------------------

class JudgeResponse(BaseModel):
    score: float
    reasoning: str
    passed: bool
    issues: List[str] = []

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"Score must be 1.0–5.0, got {v}")
        return round(v, 1)


# -------------------------------------------------------
# Mock clients — simulate different failure scenarios
# -------------------------------------------------------

class FlakyMockClient:
    """Fails on first call, succeeds on second — simulates real judge failures."""

    def __init__(self, fail_times: int = 1, good_response: str = None):
        self.call_count = 0
        self.fail_times = fail_times
        self.good_response = good_response or json.dumps({
            "score": 4.2,
            "reasoning": "Correct and well-explained.",
            "passed": True,
            "issues": []
        })

    async def call(self, prompt: str, system: str = "", temperature: float = 0.0):
        self.call_count += 1

        class Result:
            pass

        r = Result()

        if self.call_count <= self.fail_times:
            # Simulate common judge failures
            failures = [
                "Here is the score: 4.2 out of 5. The answer is good.",   # plain text
                '```json\n{"score": 9.0, "reasoning": "great", "passed": true}\n```',  # bad score
                '{"reasoning": "Good answer", "passed": true}',            # missing score
            ]
            r.text = failures[(self.call_count - 1) % len(failures)]
        else:
            r.text = self.good_response

        return r


class TemperatureMockClient:
    """Simulates score variance at temperature > 0."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
        import random
        self.random = random

    async def call(self, prompt: str, system: str = "", temperature: float = 0.0):
        class Result:
            pass

        r = Result()
        base_score = 4.0

        if temperature == 0.0:
            # Deterministic — always same score
            score = base_score
        else:
            # Adds random noise proportional to temperature
            noise = self.random.uniform(-temperature, temperature)
            score = round(max(1.0, min(5.0, base_score + noise)), 1)

        r.text = json.dumps({
            "score": score,
            "reasoning": f"Scored with temperature={temperature}",
            "passed": score >= 3.0,
            "issues": []
        })
        return r


# -------------------------------------------------------
# Parser
# -------------------------------------------------------

def parse_judge_output(raw: str) -> JudgeResponse:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(clean)
        return JudgeResponse(**data)
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        raise ValueError(f"Parse failed: {e}") from e


# -------------------------------------------------------
# DEMO 1 — Retry with error feedback
# -------------------------------------------------------

async def judge_with_retry(
    topic: str,
    expected: str,
    model_output: str,
    client,
    max_retries: int = 3
) -> JudgeResponse:
    """
    Calls the judge with retry logic.
    On each failure, tells the judge exactly what went wrong.
    """
    SYSTEM = """You are an eval judge. Score outputs 1.0-5.0.
Return only JSON: {"score": <float>, "reasoning": "<str>", "passed": <bool>, "issues": []}"""

    prompt = f"""Topic: {topic}
Expected: {expected}
Output: {model_output}
Return JSON score."""

    last_error = None

    for attempt in range(max_retries):
        result = await client.call(
            prompt=prompt,
            system=SYSTEM,
            temperature=0.0
        )

        try:
            return parse_judge_output(result.text)
        except ValueError as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {str(e)[:60]}")
            if attempt < max_retries - 1:
                # Inform the judge what went wrong — critical for retry success
                prompt += f"\n\nYour previous response failed: {e}\nReturn ONLY valid JSON."

    raise ValueError(f"Judge failed after {max_retries} attempts: {last_error}")


async def demo_retry_logic():
    print("\n--- DEMO 1: Retry With Error Feedback ---")

    # Client that fails twice then succeeds
    client = FlakyMockClient(fail_times=2)

    print("Running judge with retry (fails 2x then succeeds):")
    try:
        result = await judge_with_retry(
            topic="async/await",
            expected="async def marks a coroutine",
            model_output="async def makes a function a coroutine",
            client=client
        )
        print(f"\n  ✅ Succeeded on attempt {client.call_count}")
        print(f"  Score    : {result.score}")
        print(f"  Reasoning: {result.reasoning}")
    except ValueError as e:
        print(f"  ❌ All retries failed: {e}")

    print(f"\n  Total API calls made: {client.call_count}")
    print("""
  Why error feedback on retry works:
    Attempt 1: judge returns plain text → fails
    Retry prompt includes: "Your response failed: Parse failed..."
    Attempt 2: judge sees the error, tries to fix it
    Most failures are one-off mistakes — second attempt almost always succeeds
""")


# -------------------------------------------------------
# DEMO 2 — Temperature 0 vs temperature > 0
# -------------------------------------------------------

async def demo_temperature_consistency():
    print("\n--- DEMO 2: Temperature 0 vs Temperature > 0 ---")

    prompt = "Score: 'async def marks a coroutine'"
    runs = 5

    # temperature=0.0 — deterministic
    client_det = TemperatureMockClient(temperature=0.0)
    det_scores = []
    for _ in range(runs):
        r = await client_det.call(prompt=prompt, temperature=0.0)
        data = json.loads(r.text)
        det_scores.append(data["score"])

    # temperature=0.7 — variable
    client_var = TemperatureMockClient(temperature=0.7)
    var_scores = []
    for _ in range(runs):
        r = await client_var.call(prompt=prompt, temperature=0.7)
        data = json.loads(r.text)
        var_scores.append(data["score"])

    print(f"  temperature=0.0 scores over {runs} runs: {det_scores}")
    print(f"  Variance: {max(det_scores) - min(det_scores):.1f} ← always zero")
    print()
    print(f"  temperature=0.7 scores over {runs} runs: {var_scores}")
    print(f"  Variance: {max(var_scores) - min(var_scores):.1f} ← unpredictable")
    print("""
  Why this matters for EvalForge:
    If judge uses temperature=0.7 and scores 3.5 on Monday and 4.0 on Tuesday
    for the SAME output — that is NOT a regression.
    That is temperature noise. You cannot trust the diff.

    temperature=0.0 eliminates all noise.
    Every score difference between runs is a real change.
""")


# -------------------------------------------------------
# DEMO 3 — Measuring judge consistency
# -------------------------------------------------------

async def demo_judge_consistency_check():
    print("\n--- DEMO 3: Measuring Judge Consistency ---")
    print("Run same output through judge multiple times — scores should be identical\n")

    class ConsistentMockClient:
        async def call(self, prompt, system="", temperature=0.0):
            class R:
                text = '{"score": 4.0, "reasoning": "Consistently correct", "passed": true, "issues": []}'
            return R()

    client = ConsistentMockClient()
    test_outputs = [
        "async def marks a coroutine function",
        "async runs threads in parallel",
        "async is for slow operations",
    ]

    print("  Output | Run1 | Run2 | Run3 | Consistent?")
    print("  " + "-" * 55)

    for output in test_outputs:
        scores = []
        for _ in range(3):
            r = await client.call(f"Score: '{output}'", temperature=0.0)
            data = json.loads(r.text)
            scores.append(data["score"])
        consistent = len(set(scores)) == 1
        print(f"  '{output[:30]}...' | {scores[0]} | {scores[1]} | {scores[2]} | {'✅' if consistent else '❌'}")

    print("""
  In real usage with temperature=0.0:
    All three runs should return identical scores.
    Any difference = judge prompt needs improvement.
    Run this check before trusting your judge for regression testing.
""")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("RETRY LOGIC AND JUDGE CONSISTENCY")
    print("=" * 55)

    await demo_retry_logic()
    await demo_temperature_consistency()
    await demo_judge_consistency_check()

    print("=" * 55)
    print("Key takeaways:")
    print("  1. Always retry — judges fail on first attempt more than you'd expect")
    print("  2. Include the error in the retry prompt — judge learns from its mistake")
    print("  3. temperature=0.0 always — score variance is silent poison in eval pipelines")
    print("  4. Test your judge's consistency before shipping — run same case 3x")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())