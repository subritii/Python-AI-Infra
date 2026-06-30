# src/models.py
# Shared models used across all test files.
# In real EvalForge these live in evalforge/models.py

from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional


# --- Dataclass models (internal, you control the data) ---

@dataclass
class EvalResult:
    test_id: str
    score: float
    passed: bool
    notes: str = ""
    tags: list = field(default_factory=list)


@dataclass
class EvalRun:
    run_id: str
    model_version: str
    results: list = field(default_factory=list)
    total_cost: float = 0.0


# --- Pydantic models (external data, LLM outputs) ---

class EvalScore(BaseModel):
    score: float
    reasoning: str = ""
    passed: bool = False
    suggestions: List[str] = []

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"Score must be 1–5, got {v}")
        return round(v, 2)

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty_on_high_score(cls, v):
        return v.strip()


# --- Helper functions under test ---

def is_passing(score: float) -> bool:
    """A score of 3.0 or above is a passing eval result."""
    return score >= 3.0


def compute_run_stats(run: EvalRun) -> dict:
    """Compute pass rate and average score for a run."""
    if not run.results:
        return {"pass_rate": 0.0, "avg_score": 0.0, "total": 0}

    total = len(run.results)
    passed = sum(1 for r in run.results if r.passed)
    avg = sum(r.score for r in run.results) / total

    return {
        "pass_rate": round(passed / total, 2),
        "avg_score": round(avg, 2),
        "total": total,
    }


# --- Fake async functions (simulate Anthropic API) ---

async def fake_score_output(prompt: str, expected: str) -> EvalScore:
    """Simulates calling Claude and getting a score back.
    In real EvalForge this calls the Anthropic API."""
    return EvalScore(
        score=4.2,
        reasoning="Clear and accurate answer",
        passed=True
    )


async def fake_run_eval(test_id: str, prompt: str) -> EvalResult:
    """Simulates running one eval case through the pipeline."""
    score = 4.2
    return EvalResult(
        test_id=test_id,
        score=score,
        passed=is_passing(score)
    )
