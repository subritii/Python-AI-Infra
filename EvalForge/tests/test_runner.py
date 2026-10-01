import pytest
from evalforge import runner
from evalforge.config import Config, ConfigError, config
from evalforge.models import load_test_cases
from evalforge.runner import run_all


def test_demo_mode_is_offline():
    assert config.mock_mode and config.provider == "mock"
    assert not config.use_database


@pytest.mark.parametrize("path", ["test_cases/phase3.yaml", "test_cases/meridian_advisor.yaml"])
async def test_demo_run_end_to_end(path):
    cases = load_test_cases(path)
    run = await run_all(cases)
    assert len(run.results) == len(cases)
    assert all(r.error is None for r in run.results)
    assert run.pass_rate == 1.0 and run.avg_score == 4.2
    assert run.total_cost > 0  # token usage is recorded on results


async def test_model_call_failure_does_not_abort_batch(monkeypatch):
    cases = load_test_cases("test_cases/meridian_advisor.yaml")
    real  = runner.get_model_output

    async def flaky(tc):
        if tc.id == cases[0].id:
            raise RuntimeError("provider down")
        return await real(tc)

    monkeypatch.setattr(runner, "get_model_output", flaky)
    run = await run_all(cases)
    errored = [r for r in run.results if r.error]
    assert [r.test_id for r in errored] == [cases[0].id]
    assert "provider down" in errored[0].error
    assert run.pass_rate == 1.0  # errored results are excluded from stats


def test_bad_provider_rejected(monkeypatch):
    monkeypatch.setenv("PROVIDER", "openai")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_live_mode_requires_api_key(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("MODEL", "")
    live = Config.from_env()
    assert live.model == "openai/gpt-oss-120b"
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        live.validate_live()
