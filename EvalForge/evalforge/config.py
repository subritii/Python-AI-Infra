import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ("mock", "groq", "anthropic")

DEFAULT_MODELS = {
    "mock": "demo",
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-sonnet-4-6",
}

# The judge defaults to a different model family than the model under test, so the
# model never grades its own answers (LLM judges tend to favour their own outputs).
DEFAULT_JUDGE_MODELS = {
    "mock": "demo-judge",
    "groq": "qwen/qwen3.8-27b",
    "anthropic": "claude-sonnet-4-6",
}

API_KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    groq_api_key: str
    database_url: str
    mock_mode: bool
    model: str
    baseline_run_id: str
    max_concurrent_calls: int
    judge_temperature: float
    provider: str = "mock"
    judge_provider: str = "mock"
    judge_model: str = "demo-judge"

    @property
    def use_database(self) -> bool:
        # Demo (mock) runs never touch the database, so they stay offline-safe
        # and can't pollute the real run history / dashboard.
        return bool(self.database_url) and not self.mock_mode

    @classmethod
    def from_env(cls) -> "Config":
        mock_mode = os.getenv("MOCK_MODE", "true").strip().lower() == "true"
        provider  = os.getenv("PROVIDER", "mock").strip().lower()
        if provider not in PROVIDERS:
            raise ConfigError(f"PROVIDER must be one of {PROVIDERS}, got {provider!r}")
        if mock_mode or provider == "mock":
            mock_mode, provider = True, "mock"

        judge_provider = os.getenv("JUDGE_PROVIDER", "").strip().lower() or provider
        if judge_provider not in PROVIDERS:
            raise ConfigError(f"JUDGE_PROVIDER must be one of {PROVIDERS}, got {judge_provider!r}")
        if mock_mode:
            judge_provider = "mock"

        try:
            max_concurrent_calls = int(os.getenv("MAX_CONCURRENT_CALLS", "3"))
            judge_temperature    = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
        except ValueError as e:
            raise ConfigError(f"Invalid numeric setting: {e}") from e
        if max_concurrent_calls < 1:
            raise ConfigError("MAX_CONCURRENT_CALLS must be >= 1")

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            database_url=os.getenv("DATABASE_URL", ""),
            mock_mode=mock_mode,
            model=os.getenv("MODEL") or DEFAULT_MODELS[provider],
            baseline_run_id=os.getenv("BASELINE_RUN_ID") or None,
            max_concurrent_calls=max_concurrent_calls,
            judge_temperature=judge_temperature,
            provider=provider,
            judge_provider=judge_provider,
            judge_model=os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODELS[judge_provider],
        )

    def validate_live(self) -> None:
        """Fail fast with a clear message before any live run starts."""
        if self.mock_mode:
            return
        for role, provider in (("PROVIDER", self.provider), ("JUDGE_PROVIDER", self.judge_provider)):
            if provider == "mock":
                raise ConfigError(f"{role}=mock is only valid with MOCK_MODE=true")
            api_key = self.groq_api_key if provider == "groq" else self.anthropic_api_key
            if not api_key:
                raise ConfigError(f"{API_KEY_VARS[provider]} is required when {role}={provider} and MOCK_MODE=false")
        if not self.database_url:
            raise ConfigError("DATABASE_URL is required for live runs (results are stored in Postgres)")


config = Config.from_env()
