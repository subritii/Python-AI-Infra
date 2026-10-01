# EvalForge

A regression-testing pipeline for LLM applications. EvalForge runs a suite of
YAML test cases against a model, scores each answer with an **LLM-as-a-judge**,
stores the run in Postgres, and **fails CI when any test's score drops ≥ 1.0
versus a baseline run**.

The included suite tests *Meridian*, a fictional fintech assistant, on
compliance-critical behaviour: risk disclosures, FDIC vs. SIPC coverage, and
refusing personalized buy/sell advice.

```
YAML test cases → model call → judge call → EvalRun → Postgres → dashboard JSON
                   (bounded concurrency)   (validated JSON, retries)   (regression gate)
```

## Quick start (demo mode: offline, no keys, no database)

```bash
cd EvalForge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q          # offline unit tests
python run_regression.py     # regression suite (exit 0 = pass, 1 = regression/error)
python main.py               # general test suite
```

Demo mode is the default. A deterministic mock client stands in for both the
model and the judge, so the whole pipeline runs end to end with nothing to
configure. Demo runs never touch the database.

## Live mode

```bash
cp .env.example .env         # then set MOCK_MODE=false, PROVIDER, keys, DATABASE_URL
psql "$DATABASE_URL" -f schema.sql
python run_regression.py     # first run with no BASELINE_RUN_ID becomes the baseline
```

| Variable | Default | Notes |
|---|---|---|
| `MOCK_MODE` | `true` | `false` enables live calls |
| `PROVIDER` | `mock` | `groq` or `anthropic` |
| `MODEL` | per provider | groq: `openai/gpt-oss-120b`; anthropic: `claude-sonnet-4-6` |
| `GROQ_API_KEY` / `ANTHROPIC_API_KEY` | | required for the chosen provider |
| `DATABASE_URL` | | Postgres; required for live runs |
| `BASELINE_RUN_ID` | | run to compare against |
| `MAX_CONCURRENT_CALLS` | `3` | semaphore bound on in-flight test cases |
| `JUDGE_TEMPERATURE` | `0.0` | |

Configuration is validated at startup. An unknown provider, a missing key or a
missing database fails fast with a clear message.

## Design notes

- **Judge robustness:** judge output is parsed into a Pydantic model (score
  1.0–5.0, non-empty reasoning). Invalid output is retried up to 3 times, with
  the parse error fed back into the prompt.
- **Failure isolation:** a failed model or judge call is recorded on that test
  case and doesn't abort the batch. Any errored case fails the run and the run
  isn't saved, so a broken provider can never show up as "no regressions".
- **CI:** `.github/workflows/eval.yml` runs the unit tests, then the offline
  demo, then the live regression against Groq. On pushes to `main` it commits
  the refreshed `docs/dashboard_data.json`.

## Layout

```
evalforge/   config, client (mock/groq/anthropic), runner, scorer, storage, models
test_cases/  YAML suites
tests/       offline pytest suite
schema.sql   Postgres tables for live mode
```
