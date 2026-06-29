from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator
from typing import List, Optional
import yaml


@dataclass
class TestCase:
    id: str
    topic: str
    prompt: str
    expected_output: str
    metric: str = "llm_judge"
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
    prompt_hash: str = ""
    results: list = field(default_factory=list)
    total_cost: float = 0.0
    pass_rate: float = 0.0
    avg_score: float = 0.0

    def compute_stats(self):                                          
        valid = [r for r in self.results if r.error is None]        
        if not valid:                                                 
            return                                                    
        self.total_cost = round(sum(r.cost_usd for r in valid), 6)   
        self.pass_rate  = round(sum(1 for r in valid if r.passed) / len(valid), 2)
        self.avg_score  = round(sum(r.score for r in valid) / len(valid), 2)


class JudgeResponse(BaseModel):
    score: float
    reasoning: str
    passed: bool
    issues: List[str] = []

    @field_validator("score")                  # ← inside JudgeResponse
    @classmethod                               # ← inside JudgeResponse
    def score_in_range(cls, v):               # ← inside JudgeResponse
        if not 1.0 <= v <= 5.0:              # ← inside score_in_range
            raise ValueError(f"Score must be 1.0-5.0, got {v}")
        return round(v, 1)

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Reasoning too short")
        return v.strip()
    

def load_test_cases(path: str) -> list:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return [TestCase(**tc) for tc in raw]