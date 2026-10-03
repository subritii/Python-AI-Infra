import json
import pytest
from evalforge import scorer
from evalforge.client import APIResponse, EvalForgeClient
from evalforge.models import TestCase as Case
from evalforge.scorer import parse_judge_output, score_output


def make_case(metric="llm_judge", expected="coroutine"):
    return Case(id="t1", topic="async", prompt="What is async?",
                expected_output=expected, metric=metric)


def fake_response(text):
    return APIResponse(text=text, input_tokens=1, output_tokens=1,
                       cost_usd=0.0, model="fake", stop_reason="stop")


def test_parse_valid_json():
    r = parse_judge_output('{"score": 4.0, "reasoning": "Looks correct", "passed": true}')
    assert r.score == 4.0 and r.passed and r.issues == []


def test_parse_strips_markdown_fences():
    raw = '```json\n{"score": 3.5, "reasoning": "Partially right", "passed": true}\n```'
    assert parse_judge_output(raw).score == 3.5


@pytest.mark.parametrize("raw", [
    "not json",
    '{"score": 7.0, "reasoning": "Out of range", "passed": true}',
    '{"score": 4.0, "reasoning": "", "passed": true}',
    '{"reasoning": "Missing score", "passed": true}',
])
def test_parse_rejects_bad_judge_output(raw):
    with pytest.raises(ValueError):
        parse_judge_output(raw)


async def test_exact_metric_is_case_insensitive():
    r = await score_output(make_case("exact", "Paris"), "  paris ")
    assert r.passed and r.score == 5.0
    r = await score_output(make_case("exact", "Paris"), "London")
    assert not r.passed and r.score == 1.0


async def test_judge_retries_then_succeeds(monkeypatch):
    replies = iter(["garbage", json.dumps({"score": 4.5, "reasoning": "Good answer", "passed": True})])

    async def fake_call(**kwargs):
        return fake_response(next(replies))

    monkeypatch.setattr(scorer.judge_client, "call", fake_call)
    r = await score_output(make_case(), "a coroutine")
    assert r.error is None and r.score == 4.5


async def test_judge_failure_is_recorded_not_raised(monkeypatch):
    async def fake_call(**kwargs):
        return fake_response("still not json")

    monkeypatch.setattr(scorer.judge_client, "call", fake_call)
    r = await score_output(make_case(), "a coroutine")
    assert r.error and "3 attempts" in r.error and not r.passed


async def test_rate_limited_call_waits_and_retries(monkeypatch):

    class RateLimited(Exception):
        status_code = 429

    c = EvalForgeClient("groq", "m")
    c.mock_mode = False
    calls, sleeps = [], []

    async def fake_groq(*args):
        calls.append(1)
        if len(calls) < 3:
            raise RateLimited("Rate limit reached. Please try again in 2.5s.")
        return fake_response("ok")

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(c, "_groq_call", fake_groq)
    monkeypatch.setattr("evalforge.client.asyncio.sleep", fake_sleep)
    r = await c.call("hi")
    assert r.text == "ok" and len(calls) == 3 and sleeps == [3.5, 3.5]


@pytest.mark.parametrize("msg, seconds", [
    ("Please try again in 30.754s.", 31.754),
    ("Please try again in 1m16.896s.", 77.896),
    ("Please try again in 2h3m4s.", 7385.0),
    ("no hint here", 60.0),
])
def test_retry_after_parses_groq_waits(msg, seconds):
    assert EvalForgeClient._retry_after(Exception(msg)) == pytest.approx(seconds)


async def test_daily_quota_fails_fast(monkeypatch):
    class RateLimited(Exception):
        status_code = 429

    c = EvalForgeClient("groq", "m")
    c.mock_mode = False
    sleeps = []

    async def fake_groq(*args):
        raise RateLimited("Rate limit reached on tokens per day (TPD). Please try again in 1m16.896s.")

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(c, "_groq_call", fake_groq)
    monkeypatch.setattr("evalforge.client.asyncio.sleep", fake_sleep)
    with pytest.raises(RuntimeError, match="daily quota"):
        await c.call("hi")
    assert sleeps == []
