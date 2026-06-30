# Pydantic Output Parsing
# Converts raw LLM string output into a validated, typed Python object.
#
# LLMs return unstructured strings — they can include markdown fences, missing fields,
# or wrong types. This pattern defines the expected shape as a Pydantic model and
# forces all output through it, so bad responses fail loudly instead of silently
# corrupting downstream eval results.

import json
from pydantic import BaseModel, ValidationError
from typing import List

# Defines the exact JSON shape expected from the LLM judge
class JudgeResponse(BaseModel):
    score: float
    reasoning: str
    passed: bool
    suggestions: List[str] = []  # optional — LLM may omit this

def parse_judge_output(raw_llm_output: str) -> JudgeResponse:
    try:
        clean = raw_llm_output.strip()

        # Claude sometimes wraps JSON in ```json ... ``` markdown fences — strip them
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]

        data = json.loads(clean)
        return JudgeResponse(**data)  # Pydantic validates types and required fields here

    except json.JSONDecodeError:
        # LLM returned something that isn't JSON at all
        raise ValueError(f"LLM returned invalid JSON: {raw_llm_output[:100]}")
    except ValidationError as e:
        # Valid JSON but wrong shape — missing fields or wrong types
        raise ValueError(f"LLM output missing required fields: {e}")

raw = '{"score": 4.2, "reasoning": "Clear explanation...", "passed": true}'
result = parse_judge_output(raw)
print(result.score)    # 4.2
print(result.passed)   # True