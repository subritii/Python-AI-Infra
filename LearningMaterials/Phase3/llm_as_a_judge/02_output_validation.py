# TOPIC: Validating judge output with Pydantic
#
# What you'll learn:
#   - Why you never trust raw LLM output
#   - The JudgeResponse Pydantic model
#   - Parsing JSON with markdown fence stripping
#   - What happens when validation fails
#
# -------------------------------------------------------

import json
from pydantic import BaseModel, field_validator, ValidationError
from typing import List


# -------------------------------------------------------
# The JudgeResponse model
# This is the shape every judge response must match
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

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Reasoning too short — judge gave a useless explanation")
        return v.strip()


# -------------------------------------------------------
# The parser — strips fences and validates
# -------------------------------------------------------

def parse_judge_output(raw: str) -> JudgeResponse:
    """
    Cleans raw LLM output and validates it against JudgeResponse.

    Claude sometimes wraps JSON in markdown fences:
      ```json
      {"score": 4.2, ...}
      ```
    json.loads() cannot handle the backticks — strip them first.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned invalid JSON: {raw[:100]}") from e

    try:
        return JudgeResponse(**data)
    except ValidationError as e:
        raise ValueError(f"Judge output missing required fields: {e}") from e


# -------------------------------------------------------
# DEMO 1 — Valid responses
# -------------------------------------------------------

def demo_valid_responses():
    print("\n--- DEMO 1: Valid Judge Responses ---")

    valid_cases = [
        # Perfect clean JSON
        ('{"score": 4.5, "reasoning": "Correct and adds useful detail", "passed": true, "issues": []}',
         "Clean JSON"),

        # With markdown fences — common Claude behavior
        ('```json\n{"score": 3.0, "reasoning": "Core concept correct", "passed": true, "issues": []}\n```',
         "JSON in markdown fences"),

        # With issues list
        ('{"score": 2.0, "reasoning": "Wrong mechanism described", "passed": false, "issues": ["confuses threads with coroutines"]}',
         "With issues"),

        # Score as string — Pydantic coerces it to float
        ('{"score": "4.0", "reasoning": "Good answer overall", "passed": true, "issues": []}',
         "Score as string (Pydantic coerces)"),
    ]

    for raw, label in valid_cases:
        result = parse_judge_output(raw)
        print(f"  [{label}]")
        print(f"    score    : {result.score} ({type(result.score).__name__})")
        print(f"    passed   : {result.passed}")
        print(f"    reasoning: {result.reasoning[:60]}")
        print()


# -------------------------------------------------------
# DEMO 2 — Invalid responses — what Pydantic catches
# -------------------------------------------------------

def demo_invalid_responses():
    print("\n--- DEMO 2: Invalid Responses Pydantic Catches ---")

    invalid_cases = [
        # Score out of range
        ('{"score": 9.0, "reasoning": "Great answer", "passed": true, "issues": []}',
         "Score out of range (9.0)"),

        # Missing required field
        ('{"score": 4.0, "passed": true, "issues": []}',
         "Missing 'reasoning' field"),

        # Completely wrong format — plain text
        ("The output is correct and deserves a score of 4.",
         "Plain text instead of JSON"),

        # Empty reasoning
        ('{"score": 4.0, "reasoning": "ok", "passed": true, "issues": []}',
         "Reasoning too short"),

        # Missing score entirely
        ('{"reasoning": "Good answer", "passed": true, "issues": []}',
         "Missing 'score' field"),
    ]

    for raw, label in invalid_cases:
        try:
            result = parse_judge_output(raw)
            print(f"  [{label}] — ⚠️  Unexpectedly passed: {result}")
        except (ValueError, ValidationError) as e:
            print(f"  [{label}]")
            print(f"    ✅ Caught: {str(e)[:80]}")
            print()


# -------------------------------------------------------
# DEMO 3 — Why validation matters in a batch run
# -------------------------------------------------------

def demo_why_validation_matters():
    print("\n--- DEMO 3: Why Validation Matters ---")
    print("""
Without Pydantic validation:
  - Judge returns score=9.0 → stored in DB as 9.0
  - Pass rate calculation breaks (score > 5 is impossible)
  - Dashboard shows nonsense
  - You don't know until you look at the data

With Pydantic validation:
  - Judge returns score=9.0 → ValidationError raised immediately
  - Retry logic kicks in → judge gets another chance
  - If all retries fail → that test case is marked as error
  - Everything else continues running
  - You see exactly which case caused the problem

The rule: never store data that hasn't been validated.
""")

    # Simulate a batch where one result is bad
    raw_results = [
        '{"score": 4.5, "reasoning": "Correct answer", "passed": true, "issues": []}',
        '{"score": 9.0, "reasoning": "Perfect", "passed": true, "issues": []}',  # bad
        '{"score": 2.5, "reasoning": "Partially correct", "passed": false, "issues": ["missing detail"]}',
    ]

    valid = []
    errors = []

    for i, raw in enumerate(raw_results):
        try:
            result = parse_judge_output(raw)
            valid.append(result)
            print(f"  tc_{i+1:03d}: score={result.score} ✅")
        except (ValueError, ValidationError) as e:
            errors.append({"index": i, "error": str(e)})
            print(f"  tc_{i+1:03d}: ❌ validation failed — {str(e)[:60]}")

    print(f"\n  {len(valid)} valid, {len(errors)} failed — batch continues cleanly")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

def main():
    print("=" * 55)
    print("OUTPUT VALIDATION WITH PYDANTIC")
    print("=" * 55)

    demo_valid_responses()
    demo_invalid_responses()
    demo_why_validation_matters()

    print("=" * 55)
    print("Key takeaways:")
    print("  1. Always parse with parse_judge_output() — never json.loads() directly")
    print("  2. Pydantic validates score range, reasoning length, field presence")
    print("  3. Markdown fences are common — strip them before parsing")
    print("  4. Invalid results should be retried, not silently stored")
    print("=" * 55)


if __name__ == "__main__":
    main()