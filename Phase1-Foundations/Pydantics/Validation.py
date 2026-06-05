# Pydantic Validators: field_validator and model_validator
#
# In AI/eval pipelines, LLM outputs are unpredictable — a model may return a score
# out of range, an empty reasoning, or a suspiciously high score with no justification.
# Validators act as a contract: bad data raises a ValidationError immediately rather
# than silently flowing downstream and corrupting results or metrics.
#
# field_validator  — validates a single field in isolation (e.g. score range, string length)
# model_validator  — runs after all fields are set; used to enforce rules that span
#                    multiple fields (e.g. "if score is high, reasoning must be detailed")
#
# Execution order: field_validators first (per field) → model_validator last (whole object)
# You never call these methods directly — Pydantic calls them automatically on instantiation.

from pydantic import BaseModel, field_validator, model_validator

class EvalScore(BaseModel):
    topic: str
    score: float
    reasoning: str
    model_version: str

    # Runs when 'score' is set — rejects out-of-range values, normalizes valid ones
    @field_validator("score")
    @classmethod
    def score_in_range(cls, v):
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"Score must be 1–5, got {v}")
        return round(v, 2)

    # Runs when 'reasoning' is set — guards against lazy/empty LLM responses
    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Reasoning too short — LLM gave a bad response")
        return v.strip()

    # Runs after all fields are validated — checks relationship between score and reasoning
    # mode="after" means self.score and self.reasoning are already validated and available
    @model_validator(mode="after")
    def high_score_needs_reasoning(self):
        if self.score >= 4.5 and len(self.reasoning) < 50:
            raise ValueError("High scores need detailed reasoning")
        return self

# Intentionally invalid — score=6.0 triggers score_in_range and raises ValidationError
score = EvalScore(
    topic="async/await",
    score=6.0,
    reasoning="Good",
    model_version="claude-sonnet-4-6"
)