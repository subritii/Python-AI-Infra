# tests/test_5_mocking.py
# ---------------------------------------------------------------
# TOPIC: Mocking
#
# What you'll learn:
#   - MagicMock for sync dependencies
#   - AsyncMock for async dependencies (LLM API calls)
#   - patch() to replace by import path
#   - Asserting mocks were called correctly
#
# Why mock: tests should be fast, free, and offline.
# Never call the real Anthropic API in tests.
#
# Run: pytest tests/test_5_mocking.py -v
# ---------------------------------------------------------------

import pytest
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock, AsyncMock, patch, call
from models import EvalResult, EvalScore


# ---------------------------------------------------------------
# PART A — MagicMock (sync)
# Replace any synchronous object or function with a fake.
# You control what it returns.
# ---------------------------------------------------------------

def get_model_name(client) -> str:
    """Calls an external client to get the current model name."""
    return client.get_model()


def get_run_cost(client, run_id: str) -> float:
    """Fetches total cost for a run from an external service."""
    return client.fetch_cost(run_id=run_id)


def test_mock_sync_client():
    mock_client = MagicMock()
    mock_client.get_model.return_value = "claude-sonnet-4-6"

    result = get_model_name(mock_client)

    assert result == "claude-sonnet-4-6"
    mock_client.get_model.assert_called_once()          # was it called?


def test_mock_with_argument():
    mock_client = MagicMock()
    mock_client.fetch_cost.return_value = 0.0042

    cost = get_run_cost(mock_client, run_id="run_001")

    assert cost == 0.0042
    mock_client.fetch_cost.assert_called_once_with(run_id="run_001")  # called with right args?


def test_mock_called_multiple_times():
    mock_client = MagicMock()
    mock_client.fetch_cost.side_effect = [0.001, 0.002, 0.003]  # different return per call

    costs = [get_run_cost(mock_client, f"run_{i}") for i in range(3)]

    assert costs == [0.001, 0.002, 0.003]
    assert mock_client.fetch_cost.call_count == 3


def test_mock_not_called():
    mock_client = MagicMock()
    # If we never call get_model_name, the mock shouldn't be called
    mock_client.get_model.assert_not_called()


# ---------------------------------------------------------------
# PART B — AsyncMock (async)
# Same as MagicMock but for async functions.
# Use when the real function is `async def`.
# ---------------------------------------------------------------

async def call_judge_api(client, prompt: str, output: str) -> str:
    """Calls the LLM judge API and returns raw JSON string."""
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": f"{prompt}\n\nOutput: {output}"}]
    )
    return response.content[0].text


async def test_async_mock_basic():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text='{"score": 4.2, "reasoning": "Clear", "passed": true}')]
        )
    )

    result = await call_judge_api(mock_client, "Rate this output", "async is a keyword")

    assert '"score": 4.2' in result
    mock_client.messages.create.assert_called_once()


async def test_async_mock_returns_controlled_value():
    """Control exactly what the fake API returns."""
    fake_response = json.dumps({
        "score": 3.5,
        "reasoning": "Partially correct answer",
        "passed": True,
        "suggestions": ["Add more detail"]
    })

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text=fake_response)])
    )

    raw = await call_judge_api(mock_client, "Rate this", "Some output")
    parsed = json.loads(raw)

    assert parsed["score"] == 3.5
    assert parsed["passed"] is True
    assert len(parsed["suggestions"]) == 1


async def test_async_mock_raises_exception():
    """Simulate the API throwing an error."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        side_effect=Exception("API rate limit exceeded")
    )

    with pytest.raises(Exception) as exc_info:
        await call_judge_api(mock_client, "Rate this", "output")

    assert "rate limit" in str(exc_info.value)


# ---------------------------------------------------------------
# PART C — patch()
# Replaces something by its import path for the duration of a test.
# Use when you can't inject the dependency as a parameter.
# ---------------------------------------------------------------

# Simulate a module-level function that calls the API directly
def parse_llm_output(raw: str) -> EvalScore:
    """Parses raw LLM JSON into an EvalScore."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(clean)
    return EvalScore(**data)


def test_parse_clean_json():
    """No mock needed — pure function, no dependencies."""
    raw = '{"score": 4.2, "reasoning": "Good answer", "passed": true}'
    result = parse_llm_output(raw)
    assert result.score == 4.2
    assert result.passed is True


def test_parse_json_with_markdown_fences():
    """Test that markdown fences are stripped correctly."""
    raw = '```json\n{"score": 3.5, "reasoning": "OK", "passed": true}\n```'
    result = parse_llm_output(raw)
    assert result.score == 3.5


def test_parse_invalid_json_raises():
    """Test that malformed JSON raises an error."""
    with pytest.raises(json.JSONDecodeError):
        parse_llm_output("this is not json")


# ---------------------------------------------------------------
# PART D — Asserting call details
# Verify the mock was called with the right arguments
# ---------------------------------------------------------------

async def test_assert_called_with_correct_model():
    """Verify the API was called with the right model name."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text='{"score": 4.0, "reasoning": "Good", "passed": true}')]
        )
    )

    await call_judge_api(mock_client, "Rate this", "output text")

    # Check exact arguments
    mock_client.messages.create.assert_called_once_with(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Rate this\n\nOutput: output text"}]
    )


async def test_assert_call_count():
    """Verify the API was called the right number of times."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text='{"score": 4.0, "reasoning": "Good", "passed": true}')]
        )
    )

    # Call it 3 times
    for i in range(3):
        await call_judge_api(mock_client, f"prompt {i}", "output")

    assert mock_client.messages.create.call_count == 3
