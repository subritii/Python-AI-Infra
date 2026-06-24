# TOPIC: Complete EvalForge scorer
#
# What you'll learn:
#   - The full scorer that goes into evalforge/scorer.py
#   - Running multiple evals concurrently with the judge
#   - Aggregating scores into run-level stats
#   - How every piece from files 01-04 fits together
#
# -------------------------------------------------------

import json
import asyncio
import os
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------
# Data shapes
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


@dataclass
class TestCase:
    id: str
    topic: str
    prompt: str
    expected_output: str
    difficulty: int = 1
    tags: list = field(default_factory=list)


@dataclass
class EvalResult:
    test_id: str
    model_output: str
    score: float
    reasoning: str
    passed: bool
    issues: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalRun:
    run_id: str
    model_version: str
    temperature: float
    results: list = field(default_factory=list)
    total_cost: float = 0.0
    pass_rate: float = 0.0
    avg_score: float = 0.0

    def compute_stats(self):
        if not self.results:
            return
        valid = [r for r in self.results if r.error is None]
        if not valid:
            return
        self.total_cost  = sum(r.cost_usd for r in valid)
        self.pass_rate   = round(sum(1 for r in valid if r.passed) / len(valid), 2)
        self.avg_score   = round(sum(r.score for r in valid) / len(valid), 2)

    def summary(self) -> str:
        valid   = [r for r in self.results if r.error is None]
        errored = [r for r in self.results if r.error is not None]
        return (
            f"Run ID      : {self.run_id}\n"
            f"Model       : {self.model_version}\n"
            f"Temperature : {self.temperature}\n"
            f"Total cases : {len(self.results)}\n"
            f"Valid       : {len(valid)}\n"
            f"Errored     : {len(errored)}\n"
            f"Pass rate   : {self.pass_rate * 100:.0f}%\n"
            f"Avg score   : {self.avg_score:.1f}/5.0\n"
            f"Total cost  : ${self.total_cost:.6f}"
        )


# -------------------------------------------------------
# Mock client + mock model
# -------------------------------------------------------

class MockClient:
    """Simulates both the model being evaluated AND the judge."""

    MOCK_MODEL_OUTPUTS = {
        "tc_001": "async def marks a coroutine that can be awaited",
        "tc_002": "Tokens are the chunks of text LLMs process, smaller than words",
        "tc_003": "async runs functions in parallel threads",   # wrong — low score
        "tc_004": "Temperature controls how random the model's output is",
        "tc_005": "RAG retrieves relevant documents before generating an answer",
    }

    MOCK_JUDGE_SCORES = {
        "tc_001": {"score": 4.5, "reasoning": "Correct and adds useful detail", "passed": True, "issues": []},
        "tc_002": {"score": 3.5, "reasoning": "Correct but slightly imprecise", "passed": True, "issues": ["tokens can be smaller or larger than words depending on the tokenizer"]},
        "tc_003": {"score": 1.5, "reasoning": "Confuses coroutines with threading", "passed": False, "issues": ["async is single-threaded", "threads and coroutines are different"]},
        "tc_004": {"score": 4.0, "reasoning": "Correct core concept", "passed": True, "issues": []},
        "tc_005": {"score": 4.5, "reasoning": "Accurate and complete", "passed": True, "issues": []},
    }

    def __init__(self):
        self.current_test_id = None

    async def call(self, prompt: str, system: str = "", temperature: float = 0.0):
        class Result:
            pass
        await asyncio.sleep(0.05)   # simulate latency
        r = Result()

        # Detect if this is the judge call or model call
        if "Score this output" in prompt and self.current_test_id:
            r.text = json.dumps(self.MOCK_JUDGE_SCORES.get(
                self.current_test_id,
                {"score": 3.0, "reasoning": "Default mock score", "passed": True, "issues": []}
            ))
            r.input_tokens  = len(prompt) // 4
            r.output_tokens = 40
        else:
            r.text = self.MOCK_MODEL_OUTPUTS.get(self.current_test_id, "Default mock output")
            r.input_tokens  = len(prompt) // 4
            r.output_tokens = 20

        r.cost_usd = (r.input_tokens * 0.000003) + (r.output_tokens * 0.000015)
        return r


# -------------------------------------------------------
# The scorer
# -------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an eval scoring assistant for EvalForge.

Rubric:
  5.0 — Perfect: fully correct, clear, complete
  4.0 — Good: correct with minor gaps
  3.0 — Partial: right concept, missing details
  2.0 — Poor: partially relevant but misleading
  1.0 — Wrong: incorrect, off-topic, or empty

Rules:
  - Return ONLY valid JSON
  - Score in 0.5 increments
  - passed = true if score >= 3.0
  - Cap at 2.5 if output contains factual error

Schema: {"score": 4.0, "reasoning": "...", "passed": true, "issues": []}"""


def parse_judge_output(raw: str) -> JudgeResponse:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(clean)
    return JudgeResponse(**data)


async def run_eval_case(
    test_case: TestCase,
    client: MockClient,
    sem: asyncio.Semaphore,
    max_retries: int = 3
) -> EvalResult:
    """Run one test case through model + judge."""

    async with sem:
        client.current_test_id = test_case.id

        # Step 1 — Get model output
        model_result = await client.call(
            prompt=test_case.prompt,
            temperature=0.7   # higher for generation
        )
        model_output = model_result.text

        # Step 2 — Send to judge
        judge_prompt = f"""Score this output:

Topic: {test_case.topic}
Expected: {test_case.expected_output}
Output: {model_output}

Think step by step then return JSON score."""

        last_error = None
        for attempt in range(max_retries):
            try:
                judge_result = await client.call(
                    prompt=judge_prompt,
                    system=JUDGE_SYSTEM_PROMPT,
                    temperature=0.0   # always 0 for judge
                )
                scored = parse_judge_output(judge_result.text)

                return EvalResult(
                    test_id=test_case.id,
                    model_output=model_output,
                    score=scored.score,
                    reasoning=scored.reasoning,
                    passed=scored.passed,
                    issues=scored.issues,
                    input_tokens=model_result.input_tokens + judge_result.input_tokens,
                    output_tokens=model_result.output_tokens + judge_result.output_tokens,
                    cost_usd=model_result.cost_usd + judge_result.cost_usd
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    judge_prompt += f"\n\nPrevious response failed: {e}\nReturn valid JSON only."

        return EvalResult(
            test_id=test_case.id,
            model_output=model_output,
            score=0.0,
            reasoning="",
            passed=False,
            error=str(last_error)
        )


# -------------------------------------------------------
# DEMO — Full batch eval run
# -------------------------------------------------------

async def demo_full_eval_run():
    print("\n--- Full Batch Eval Run ---")

    test_cases = [
        TestCase("tc_001", "async/await",    "What does async def do?",           "async def marks a coroutine function"),
        TestCase("tc_002", "tokenization",   "What is a token in LLMs?",          "A token is a chunk of text — subword unit"),
        TestCase("tc_003", "async/await",    "How does async work internally?",   "async uses a single-threaded event loop"),
        TestCase("tc_004", "sampling",       "What does temperature control?",    "Temperature controls randomness in token sampling"),
        TestCase("tc_005", "RAG",            "What is RAG?",                      "Retrieval Augmented Generation — retrieves docs before generating"),
    ]

    client = MockClient()
    sem    = asyncio.Semaphore(3)

    print(f"Running {len(test_cases)} test cases (max 3 concurrent)...\n")

    results = await asyncio.gather(*[
        run_eval_case(tc, client, sem)
        for tc in test_cases
    ])

    run = EvalRun(
        run_id="run_001",
        model_version="mock-claude-sonnet",
        temperature=0.7,
        results=list(results)
    )
    run.compute_stats()

    print("Results:")
    print(f"  {'ID':<8} {'Score':<7} {'Pass':<6} {'Reasoning'}")
    print("  " + "-" * 60)
    for r in run.results:
        if r.error:
            print(f"  {r.test_id:<8} {'ERROR':<7} {'❌':<6} {r.error[:40]}")
        else:
            icon = "✅" if r.passed else "❌"
            print(f"  {r.test_id:<8} {r.score:<7} {icon:<6} {r.reasoning[:40]}")

    print(f"\n{run.summary()}")


# -------------------------------------------------------
# Run
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("COMPLETE EVALFORGE SCORER")
    print("=" * 55)

    await demo_full_eval_run()

    print("\n" + "=" * 55)
    print("This is your Phase 4 scorer.")
    print("Copy into evalforge/scorer.py")
    print("Swap MockClient for EvalForgeClient when ready")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())