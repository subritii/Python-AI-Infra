# Phase 3 — Anthropic SDK

## Setup

```bash
pip install anthropic python-dotenv
cp .env.example .env
```

Leave `MOCK_MODE=true` — all files run free without an API key.

## Files

| File | Topic | What to observe |
|---|---|---|
| `01_basic_call.py` | Basic API call + response shape | Print every field of the response object |
| `02_async_client.py` | Async + concurrency | Watch the time difference: sequential vs gather |
| `03_streaming.py` | Streaming tokens | See tokens arrive in real time |
| `04_error_handling.py` | Error types + backoff | Watch the retry countdown on flaky calls |
| `05_mock_client.py` | Complete EvalForge client | The actual client you'll use in Phase 4 |

## Run order

```bash
python 01_basic_call.py
python 02_async_client.py
python 03_streaming.py
python 04_error_handling.py
python 05_mock_client.py
```

## Notion tracking

After each file — update your Notion tracker:
- Status → Learned
- Confidence → 1–5
- Summary → 2–3 sentences in your own words
- Key code snippet → one pattern you'll reuse
- Project use → where this shows up in EvalForge
