import json
import pytest
from evalforge import scorer
from evalforge.client import APIResponse
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

    monkeypatch.setattr(scorer.client, "call", fake_call)
    r = await score_output(make_case(), "a coroutine")
    assert r.error is None and r.score == 4.5


async def test_judge_failure_is_recorded_not_raised(monkeypatch):
    async def fake_call(**kwargs):
        return fake_response("still not json")

    monkeypatch.setattr(scorer.client, "call", fake_call)
    r = await score_output(make_case(), "a coroutine")
    assert r.error and "3 attempts" in r.error and not r.passed
