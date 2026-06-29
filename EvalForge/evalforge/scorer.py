import json
import re
import asyncio
from pydantic import ValidationError
from evalforge.models import TestCase, EvalResult, JudgeResponse
from evalforge.client import client
from evalforge.config import config

JUDGE_SYSTEM_PROMPT = """You are an eval scoring assistant for EvalForge.

Score LLM outputs against expected answers using this rubric:
  5.0 — Perfect: fully correct, clear, complete
  4.0 — Good: correct with minor gaps
  3.0 — Partial: right concept, missing details
  2.0 — Poor: partially relevant but misleading
  1.0 — Wrong: incorrect, off-topic, or empty

Rules:
  - Return ONLY valid JSON, no markdown fences
  - Score in 0.5 increments
  - passed = true if score >= 3.0
  - Cap at 2.5 if output contains a factual error

Schema:
{"score": 4.0, "reasoning": "one sentence", "passed": true, "issues": []}

Examples:
Output: "async def marks a coroutine that can be awaited"
Expected: "async def marks a coroutine function"
Result: {"score": 4.5, "reasoning": "Correct and adds useful detail", "passed": true, "issues": []}

Output: "async runs functions in parallel threads"
Expected: "async def marks a coroutine function"
Result: {"score": 1.5, "reasoning": "Confuses coroutines with threading", "passed": false, "issues": ["async is single-threaded"]}"""


def build_judge_prompt(topic: str, expected: str, output: str) -> str:
    return f"""Score this output:

Topic: {topic}
Expected: {expected}
Output: {output}

Think step by step then return your JSON score."""

def parse_judge_output(raw: str) -> JudgeResponse:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(clean)
        return JudgeResponse(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from judge: {raw[:100]}") from e
    except ValidationError as e:
        raise ValueError(f"Judge output missing fields: {e}") from e
    
async def judge_output(
    topic: str,
    expected: str,
    model_output: str,
    max_retries: int = 3
) -> JudgeResponse:

    prompt     = build_judge_prompt(topic, expected, model_output)
    last_error = None

    for attempt in range(max_retries):
        result = await client.call(
            prompt=prompt,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=config.judge_temperature
        )
        try:
            return parse_judge_output(result.text)
        except ValueError as e:
            last_error = e
            if attempt < max_retries - 1:
                prompt += f"\n\nPrevious attempt failed: {e}\nReturn valid JSON only."

    raise ValueError(f"Judge failed after {max_retries} attempts: {last_error}")


# Use try/except to catch any unexpected errors. One failing test case does not kill the other 19. The error is recorded, the batch continues, and compute_stats() filters out errored results with r.error is None.
async def score_output(
    test_case: TestCase,
    model_output: str
) -> EvalResult:

    try:
        if test_case.metric == "exact":
            passed = model_output.strip().lower() == test_case.expected_output.strip().lower()
            score  = 5.0 if passed else 1.0
            return EvalResult(
                test_id=test_case.id,
                model_output=model_output,
                score=score,
                reasoning="Exact match check",
                passed=passed
            )

        else:  # llm_judge (default)
            judge = await judge_output(
                topic=test_case.topic,
                expected=test_case.expected_output,
                model_output=model_output
            )
            return EvalResult(
                test_id=test_case.id,
                model_output=model_output,
                score=judge.score,
                reasoning=judge.reasoning,
                passed=judge.passed,
                issues=judge.issues
            )

    except Exception as e:
        return EvalResult(
            test_id=test_case.id,
            model_output=model_output,
            score=0.0,
            reasoning="",
            passed=False,
            error=str(e)
        )