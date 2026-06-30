# TOPIC: Few-shot examples and rubric design
#
# What you'll learn:
#   - Why few-shot examples make judges consistent
#   - What a good rubric looks like vs a bad one
#   - How to cover the three cases you always need
#   - How example quality affects judge quality
#
# -------------------------------------------------------

import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------
# Mock client
# -------------------------------------------------------

class MockClient:
    def __init__(self, fixed_score: float = 4.2):
        self.fixed_score = fixed_score
        self.last_prompt = ""
        self.last_system = ""

    async def call(self, prompt: str, system: str = "", temperature: float = 0.0):
        self.last_prompt = prompt
        self.last_system = system

        class Result:
            pass

        r = Result()
        r.text = json.dumps({
            "score": self.fixed_score,
            "reasoning": "Mock reasoning for demo purposes.",
            "passed": self.fixed_score >= 3.0,
            "issues": []
        })
        return r


# -------------------------------------------------------
# DEMO 1 — Bad rubric vs good rubric
# -------------------------------------------------------

def demo_rubric_quality():
    print("\n--- DEMO 1: Bad Rubric vs Good Rubric ---")

    print("""
BAD RUBRIC (too vague — judge invents its own meaning):
  "Score from 1 to 5."

  Problems:
  ❌ What does 3 mean? What does 4 mean?
  ❌ Judge will invent different criteria each run
  ❌ Scores across runs are not comparable
  ❌ No guidance on edge cases (empty output, hallucination)


GOOD RUBRIC (explicit — judge has no ambiguity):
  5.0 — Perfect: fully correct, clear, complete, nothing wrong
  4.0 — Good: correct core with minor gaps or slight imprecision
  3.0 — Partial: right concept but missing key details
  2.0 — Poor: partially relevant but fundamentally misleading
  1.0 — Wrong: incorrect, off-topic, empty, or harmful

  Plus rules:
  - Score in 0.5 increments only
  - If output contains a confident factual error, cap at 2.5
  - passed = true if score >= 3.0

  Why this works:
  ✅ Every score level has an explicit definition
  ✅ Judge applies the same criteria every run
  ✅ Scores are comparable across model versions
  ✅ Edge cases are handled by explicit rules
""")


# -------------------------------------------------------
# DEMO 2 — The three examples you always need
# -------------------------------------------------------

def demo_three_examples():
    print("\n--- DEMO 2: The Three Examples You Always Need ---")

    print("""
You need at least three examples in your judge prompt.
They must cover the full scoring spectrum — not just good outputs.


EXAMPLE 1 — Perfect/good output (score 4.5)
  Shows the judge what a strong answer looks like.
  Output: "async def marks a coroutine that can be awaited"
  Expected: "async def marks a coroutine function"
  Result: {"score": 4.5, "reasoning": "Correct and adds useful detail", "passed": true, "issues": []}


EXAMPLE 2 — Wrong output (score 1.5)
  Shows the judge what a fundamentally wrong answer looks like.
  Most important — without this, judges are too lenient.
  Output: "async runs functions in parallel threads"
  Expected: "async def marks a coroutine function"
  Result: {"score": 1.5, "reasoning": "Confuses coroutines with threading", "passed": false, "issues": ["async is single-threaded"]}


EXAMPLE 3 — Partial output (score 2.5)
  Shows the judge the middle ground — right area, wrong specifics.
  Hardest to get right without an example.
  Output: "async is used for slow operations like network calls"
  Expected: "async def marks a coroutine function"
  Result: {"score": 2.5, "reasoning": "Correct use case, misses the definition", "passed": false, "issues": ["missing: what async def does to the function"]}


Why all three matter:
  Without example 2 → judge scores wrong answers too high
  Without example 3 → judge can't distinguish partial from wrong
  Without example 1 → judge doesn't know what perfect looks like
""")


# -------------------------------------------------------
# DEMO 3 — Complete system prompt with everything
# -------------------------------------------------------

FULL_JUDGE_SYSTEM_PROMPT = """You are an eval scoring assistant for EvalForge.

Your job: score LLM outputs against expected answers using the rubric below.

Rubric:
  5.0 — Perfect: fully correct, clear, complete, nothing wrong or missing
  4.0 — Good: correct core with minor gaps or slight imprecision
  3.0 — Partial: right concept but missing key details
  2.0 — Poor: partially relevant but fundamentally misleading
  1.0 — Wrong: incorrect, off-topic, empty, or harmful

Rules:
  - Return ONLY valid JSON — no markdown fences, no preamble
  - Score in 0.5 increments only (1.0, 1.5, 2.0 ... 5.0)
  - passed = true if score >= 3.0, false otherwise
  - If output contains a confident factual error, cap score at 2.5
  - Be consistent — the same output always gets the same score

Schema:
{
  "score": 4.5,
  "reasoning": "One sentence explaining the score",
  "passed": true,
  "issues": ["optional specific problems"]
}

Examples:

Output: "async def marks a coroutine that can be awaited"
Expected: "async def marks a coroutine function"
Result: {"score": 4.5, "reasoning": "Correct and adds useful detail about awaiting", "passed": true, "issues": []}

Output: "async runs functions in parallel threads"
Expected: "async def marks a coroutine function"
Result: {"score": 1.5, "reasoning": "Confuses coroutines with threading — wrong mechanism", "passed": false, "issues": ["async is single-threaded, not parallel", "threads and coroutines are different"]}

Output: "async is used for slow operations like network calls"
Expected: "async def marks a coroutine function"
Result: {"score": 2.5, "reasoning": "Correct use case but misses the definition entirely", "passed": false, "issues": ["missing: what async def actually does to the function"]}"""


async def demo_full_prompt():
    print("\n--- DEMO 3: Full Judge Prompt Structure ---")

    client = MockClient(fixed_score=4.2)

    prompt = f"""Score this output:

Topic: async/await
Expected: async def marks a coroutine function
Output: async def makes a function a coroutine

Think through accuracy and completeness step by step, then return your JSON score."""

    result = await client.call(
        prompt=prompt,
        system=FULL_JUDGE_SYSTEM_PROMPT,
        temperature=0.0
    )

    print("System prompt layers:")
    print("  Layer 1 — Role + job description")
    print("  Layer 2 — Rubric (what each score means)")
    print("  Layer 3 — Rules (edge cases, format, consistency)")
    print("  Layer 4 — JSON schema (exact output shape)")
    print("  Layer 5 — Three few-shot examples (perfect, wrong, partial)")
    print("\nUser message layers:")
    print("  Topic, Expected, Output")
    print("  Chain-of-thought trigger at the end")

    data = json.loads(result.text)
    print(f"\nJudge returned: score={data['score']}, passed={data['passed']}")
    print(f"  (Mock — real judge would analyze the actual output)")


# -------------------------------------------------------
# DEMO 4 — How example quality affects judge quality
# -------------------------------------------------------

def demo_example_quality():
    print("\n--- DEMO 4: Example Quality Rules ---")

    print("""
Rules for writing good few-shot examples:

1. COVER THE RANGE
   Include easy (score 5), medium (score 3), hard (score 1.5)
   ❌ Three examples all scoring 4+ → judge is too lenient

2. USE REAL DOMAIN EXAMPLES
   Examples from your actual test bank, not generic ones
   ❌ Generic examples → judge misunderstands your rubric

3. MAKE ISSUES SPECIFIC
   ❌ "issues": ["wrong answer"]
   ✅ "issues": ["claims async uses threads — it uses an event loop"]

4. LAST EXAMPLE MATTERS MOST
   Model pays most attention to the last example before the task
   → Put your most representative case last

5. EXAMPLES MUST ACTUALLY BE CORRECT
   Wrong examples teach wrong behavior
   → Review every example manually before shipping
""")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("FEW-SHOT EXAMPLES AND RUBRIC DESIGN")
    print("=" * 55)

    demo_rubric_quality()
    demo_three_examples()
    await demo_full_prompt()
    demo_example_quality()

    print("=" * 55)
    print("Key takeaways:")
    print("  1. Always define what each score level means explicitly")
    print("  2. Three examples minimum: perfect, wrong, partial")
    print("  3. Specific issues in examples teach the judge what to look for")
    print("  4. Wrong examples are the most important — without them judges are too lenient")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())