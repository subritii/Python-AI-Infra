# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two independent Python projects, each with its own `requirements.txt`:

- `EvalForge/` — an LLM evaluation / regression-testing pipeline (the main project, run in CI).
- `LearningMaterials/` — standalone study scripts (async, Pydantic, pytest, Anthropic SDK, LLM-as-a-judge). Files are self-contained exercises, not a package; `pytest.ini` there sets `asyncio_mode = auto`.

## EvalForge commands

All EvalForge commands run from inside `EvalForge/` (imports and relative paths like `test_cases/*.yaml` and `docs/dashboard_data.json` assume that cwd). Use the local venv:

```bash
cd EvalForge && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q         # offline unit tests (tests/conftest.py forces mock mode)
python main.py              # run test_cases/phase3.yaml, print results (saves to DB in live mode)
python run_regression.py    # run test_cases/meridian_advisor.yaml; in live mode compare to baseline + export dashboard JSON; exits 1 on regression or any errored case
```

Configuration comes from env vars / `EvalForge/.env` (see `evalforge/config.py`, template in `.env.example`): `MOCK_MODE` (default `true`), `PROVIDER` (`mock` | `groq` | `anthropic`), `MODEL` (defaults per provider), `DATABASE_URL` (Postgres, live mode only — schema in `schema.sql`), `BASELINE_RUN_ID`, `MAX_CONCURRENT_CALLS`, `JUDGE_TEMPERATURE`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`. Demo (mock) mode never touches the database (`config.use_database`); `config.validate_live()` fails fast on missing keys/DB.

## EvalForge architecture

Pipeline: YAML test cases → model call → judge call → `EvalRun` → Postgres → dashboard JSON.

- `config.py` builds a frozen `config` singleton at import time from env; `client.py` likewise creates a module-level `client` singleton. Changing env vars after import has no effect — set them before importing `evalforge`.
- `client.EvalForgeClient.call()` dispatches on `mock_mode` first, then `provider`. Mock mode returns a canned judge JSON when the prompt contains "score this output" (which `scorer.build_judge_prompt` always does), so the full pipeline runs offline. Groq uses the OpenAI SDK against Groq's endpoint with `config.model` (default `openai/gpt-oss-120b`; Groq retired `llama-3.3-70b-versatile`) and zero cost.
- `runner.run_eval_case()` catches model-call failures into `EvalResult.error` and copies token usage/cost onto the result. `runner.run_all()` fans out all test cases with `asyncio.gather`, bounded by a semaphore of `MAX_CONCURRENT_CALLS`. The Meridian fintech system prompt under test lives in `runner.get_model_output`.
- `scorer.py`: `metric: exact` does case-insensitive string match; `llm_judge` (default) calls the judge with `JUDGE_SYSTEM_PROMPT`, parses/validates into `models.JudgeResponse` (score 1.0–5.0), and retries up to 3 times, appending the parse error to the prompt. Any exception is captured into `EvalResult.error` rather than raised — errored results are excluded from `compute_stats()` and from DB inserts; the entry points refuse to save and exit 1 if any result errored.
- `storage.py` uses asyncpg against `eval_runs` and `eval_results` tables (schema in `EvalForge/schema.sql`, no migrations). A regression is a per-test score drop of ≥ 1.0 vs. the baseline run; that logic is duplicated in `run_regression.py` and `export_dashboard_data`.

## CI

`.github/workflows/eval.yml` runs pytest, an offline demo `run_regression.py`, then the live `python run_regression.py` from `EvalForge/` on Python 3.11 against Groq (`MODEL=openai/gpt-oss-120b`, `MOCK_MODE=false`, `MAX_CONCURRENT_CALLS=1`) on pushes and PRs to `main`. On push events only, it commits the regenerated `EvalForge/docs/dashboard_data.json` back to `main` with `[skip ci]` — expect that file to change on `main` independently of local edits.
