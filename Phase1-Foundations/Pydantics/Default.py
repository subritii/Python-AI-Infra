# Pydantic Default Values
# Demonstrates required vs optional fields and how to set defaults in a Pydantic model.
#
# How to identify optional vs required fields:
#   - Required:            topic: str              (no default — must be passed)
#   - Optional (default):  difficulty: int = 1     (has a value after '=')
#   - Optional (factory):  tags: list[str] = []    (use Field(default_factory=...) for mutables)
#   - Optional (nullable): notes: Optional[str] = None
#
# Rule: if a field has '= anything' after it, it's optional. If not, it's required.
# Runtime check: print(TestCase.model_fields) — shows required=True or the default value.

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TestCase(BaseModel):
    topic: str
    prompt: str
    expected_output: str
    difficulty: int = 1
    tags: list[str] = []
    notes: Optional[str] = None
    # Field(default_factory=...) ensures a fresh timestamp per instance, unlike datetime.now() which is evaluated once at class definition
    created_at: datetime = Field(default_factory=datetime.now)

# Only required fields are passed; optional fields fall back to their defaults
case = TestCase(
    topic="async/await",
    prompt="What does async def do?",
    expected_output="Marks a coroutine function"
)

print(case.difficulty)   # 1
print(case.notes)        # None
print(case.tags)         # []