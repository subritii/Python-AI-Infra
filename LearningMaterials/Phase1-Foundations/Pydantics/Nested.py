# Pydantic Nested Models
# Demonstrates embedding one Pydantic model inside another for structured, validated data.
#
# Why this matters in AI/eval pipelines:
#   LLM outputs are often complex — a single eval result contains scores, metadata, and
#   raw output all at once. Nested models let you validate each layer independently,
#   reuse sub-models (e.g. ScoreBreakdown) across multiple result types, and access
#   deeply structured data cleanly via dot notation (result.scores.accuracy).
#   Without this, you'd be passing around raw dicts with no type guarantees.

from pydantic import BaseModel
from typing import List

# Standalone sub-model — can be reused across different eval result types
class ScoreBreakdown(BaseModel):
    accuracy: float
    clarity: float
    completeness: float

    # @property computes on access, not stored — always reflects current score values
    @property
    def average(self) -> float:
        return round(
            (self.accuracy + self.clarity + self.completeness) / 3, 2
        )

# Parent model — scores field accepts either a ScoreBreakdown instance or a plain dict;
# Pydantic coerces and validates the dict into ScoreBreakdown automatically
class EvalResult(BaseModel):
    test_case_id: str
    model_output: str
    scores: ScoreBreakdown
    passed: bool
    latency_ms: int

result = EvalResult(
    test_case_id="tc_001",
    model_output="async marks a coroutine..",
    scores={"accuracy": 4.5, "clarity": 4.0, "completeness": 3.5},  # dict auto-coerced
    passed=True,
    latency_ms=312
)

print(result.scores.average)   # 4.0
print(result.scores.accuracy)  # 4.5