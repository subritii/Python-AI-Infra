import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    database_url: str
    mock_mode: bool
    model: str
    baseline_run_id: str
    max_concurrent_calls: int
    judge_temperature: float

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "mock-key"),
            database_url=os.getenv("DATABASE_URL", ""),
            mock_mode=os.getenv("MOCK_MODE", "true").lower() == "true",
            model=os.getenv("MODEL", "claude-sonnet-4-6"),
            baseline_run_id=os.getenv("BASELINE_RUN_ID", "run_001"),
            max_concurrent_calls=int(os.getenv("MAX_CONCURRENT_CALLS", "3")),
            judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
        )


config = Config.from_env()