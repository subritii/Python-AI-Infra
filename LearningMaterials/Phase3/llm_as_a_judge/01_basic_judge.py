# Topic: What a judge call looks like from scratch
#
# What you'll learn:
#   - The minimum viable judge prompt
#   - How to send output + expected to a judge
#   - What the judge returns and how to read it
#   - Why the prompt structure matters
#
# -------------------------------------------------------

import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


# -------------------------------------------------------
# Mock client — same interface as the real EvalForgeClient
# -------------------------------------------------------

class MockClient:
    async def call(self, prompt: str, system: str = "", temperature: float = 0.0):
        class Result:
            text = json.dumps({
                "score": 4.2,
                "reasoning": "The output correctly identifies async def as marking a coroutine with minor imprecision.",
                "passed": True,
                "issues": []
            })
        return Result()


# -------------------------------------------------------
# DEMO 1 — The minimum viable judge prompt (no rubric)
# Shows why a bare prompt is not enough
# -------------------------------------------------------

async def demo_bare_prompt():
    print("\n--- DEMO 1: Bare Judge Prompt (bad practice) ---")

    client = MockClient()

    # This is what most beginners write first — too vague
    bare_prompt = """Score this output from 1 to 5.

Output: "async def marks a coroutine that can be awaited"
Expected: "async def marks a coroutine function"

Return JSON with score and reasoning."""

    result = await client.call(prompt=bare_prompt, temperature=0.0)
    print(f"Prompt sent:\n{bare_prompt}")
    print(f"\nResponse:\n{result.text}")
    print("""
Problems with this prompt:
  No rubric — judge decides what 1-5 means on its own
  No output schema — judge might return {"Score": 4} not {"score": 4.0}
  No rules — judge doesn't know what to do with edge cases
  No few-shot examples — judge has no style reference
    """)


# -------------------------------------------------------
# DEMO 2 — A proper judge prompt (with rubric + schema)
# -------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an eval scoring assistant for EvalForge.

Your job: score LLM outputs against expected answers using the rubric below.

Rubric:
  5.0 — Perfect: fully correct, clear, nothing missing
  4.0 — Good: correct with minor gaps or slight imprecision
  3.0 — Partial: core concept right but missing key details
  2.0 — Poor: partially relevant but fundamentally wrong
  1.0 — Wrong: incorrect, off-topic, or empty

Rules:
  - Return ONLY valid JSON — no markdown fences, no extra text
  - Score in 0.5 increments only (1.0, 1.5, 2.0 ... 5.0)
  - passed = true if score >= 3.0, false otherwise
  - If output contains a confident factual error, cap score at 2.5

Schema:
{
  "score": 4.5,
  "reasoning": "One sentence explaining the score",
  "passed": true,
  "issues": ["optional list of specific problems found"]
}"""


def build_judge_prompt(topic: str, expected: str, output: str) -> str:
    return f"""Score this output:

Topic: {topic}
Expected: {expected}
Output: {output}

Think through accuracy and completeness step by step, then return your JSON score."""


async def demo_proper_prompt():
    print("\n--- DEMO 2: Proper Judge Prompt (good practice) ---")

    client = MockClient()

    prompt = build_judge_prompt(
        topic="async/await",
        expected="async def marks a coroutine function",
        output="async def marks a coroutine that can be awaited"
    )

    result = await client.call(
        prompt=prompt,
        system=JUDGE_SYSTEM_PROMPT,
        temperature=0.0
    )

    data = json.loads(result.text)
    print(f"Score     : {data['score']}")
    print(f"Reasoning : {data['reasoning']}")
    print(f"Passed    : {data['passed']}")
    print(f"Issues    : {data['issues']}")
    print("""
What makes this better:
  ✅ Explicit rubric — judge knows what each score means
  ✅ Exact JSON schema — consistent field names and types
  ✅ Clear rules — handles edge cases (factual errors, empty output)
  ✅ Chain-of-thought trigger — "think step by step" before scoring
    """)


# -------------------------------------------------------
# DEMO 3 — Score different outputs for the same question
# Shows the spectrum from wrong to perfect
# -------------------------------------------------------

async def demo_score_spectrum():
    print("\n--- DEMO 3: Scoring Spectrum ---")
    print("Same expected answer, four different output quality levels\n")

    expected = "async def marks a coroutine function"
    topic = "async/await"

    test_outputs = [
        ("Perfect",  "async def marks a coroutine function that can be awaited",    5.0),
        ("Good",     "async def makes a function a coroutine",                       4.0),
        ("Partial",  "async is used for slow operations like network calls",         2.5),
        ("Wrong",    "async runs functions in parallel using threads",               1.5),
    ]

    client = MockClient()

    for label, output, expected_score in test_outputs:
        prompt = build_judge_prompt(topic, expected, output)
        # Mock returns 4.2 always — in real mode this would differ
        print(f"  [{label}] Output: '{output[:60]}'")
        print(f"           Expected score: {expected_score}")
        print(f"           (Real judge would return ~{expected_score})\n")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("BASIC JUDGE CALL")
    print("=" * 55)
    print(f"MOCK_MODE: {MOCK_MODE}")

    await demo_bare_prompt()
    await demo_proper_prompt()
    await demo_score_spectrum()

    print("=" * 55)
    print("Key takeaways:")
    print("  1. A bare prompt is not enough — always include a rubric")
    print("  2. Always specify the exact JSON schema in the system prompt")
    print("  3. temperature=0.0 — always, for consistency")
    print("  4. Chain-of-thought at end of user message improves accuracy")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())