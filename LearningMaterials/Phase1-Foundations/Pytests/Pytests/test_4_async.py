# tests/test_4_async.py
# ---------------------------------------------------------------
# TOPIC: Async tests
#
# What you'll learn:
#   - Writing async test functions
#   - Async fixtures
#   - Running async tests with pytest-asyncio
#
# Requirements:
#   pip install pytest-asyncio
#
# pytest.ini must contain:
#   [pytest]
#   asyncio_mode = auto
#
# Run: pytest tests/test_4_async.py -v
# ---------------------------------------------------------------

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import EvalResult, EvalScore, fake_score_output, fake_run_eval, is_passing


# ---------------------------------------------------------------
# PART A — Basic async tests
# Write async def test_* exactly like normal tests.
# pytest-asyncio handles the event loop automatically.
# ---------------------------------------------------------------

async def test_fake_score_returns_result():
    """Await the async function — result comes back like normal."""
    result = await fake_score_output(
        prompt="What does async def do?",
        expected="Marks a coroutine function"
    )
    assert result.score == 4.2


async def test_fake_score_is_passing():
    result = await fake_score_output("any prompt", "any expected")
    assert result.passed is True


async def test_fake_score_within_range():
    result = await fake_score_output("any prompt", "any expected")
    assert 1.0 <= result.score <= 5.0


async def test_fake_score_has_reasoning():
    result = await fake_score_output("any prompt", "any expected")
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


# ---------------------------------------------------------------
# PART B — Async fixtures
# Fixtures used by async tests work normally.
# Just make sure asyncio_mode = auto in pytest.ini
# ---------------------------------------------------------------

@pytest.fixture
def sample_prompt():
    """Regular (sync) fixture works fine with async tests."""
    return {
        "prompt": "Explain what a Semaphore does in asyncio",
        "expected": "Limits the number of concurrent coroutines"
    }


@pytest.fixture
def expected_score_range():
    return {"min": 1.0, "max": 5.0}


async def test_async_with_sync_fixture(sample_prompt):
    """Async test using a regular fixture."""
    result = await fake_score_output(
        prompt=sample_prompt["prompt"],
        expected=sample_prompt["expected"]
    )
    assert result.score >= 1.0


async def test_score_in_range(expected_score_range):
    result = await fake_score_output("prompt", "expected")
    assert expected_score_range["min"] <= result.score <= expected_score_range["max"]


# ---------------------------------------------------------------
# PART C — Running multiple async calls
# asyncio.gather runs coroutines concurrently — like your eval runner
# ---------------------------------------------------------------

async def test_multiple_concurrent_evals():
    """Simulates running multiple evals concurrently with gather."""
    test_cases = [
        ("tc_001", "What is async?"),
        ("tc_002", "What is a token?"),
        ("tc_003", "What is RAG?"),
    ]

    # Run all three concurrently — this is the pattern in your eval runner
    results = await asyncio.gather(*[
        fake_run_eval(test_id, prompt)
        for test_id, prompt in test_cases
    ])

    assert len(results) == 3
    for result in results:
        assert isinstance(result, EvalResult)
        assert 1.0 <= result.score <= 5.0


async def test_gather_preserves_order():
    """asyncio.gather returns results in the same order as inputs."""
    ids = ["tc_001", "tc_002", "tc_003"]

    results = await asyncio.gather(*[
        fake_run_eval(test_id, "any prompt")
        for test_id in ids
    ])

    # Results come back in the same order
    for i, result in enumerate(results):
        assert result.test_id == ids[i]


# ---------------------------------------------------------------
# PART D — Semaphore pattern (rate limiting)
# This is exactly how your EvalForge runner limits concurrent API calls
# ---------------------------------------------------------------

async def test_semaphore_limits_concurrency():
    """Demonstrates the Semaphore pattern used in the eval runner."""
    sem = asyncio.Semaphore(2)      # max 2 concurrent calls at once
    results = []

    async def bounded_eval(test_id: str):
        async with sem:             # blocks if 2 are already running
            result = await fake_run_eval(test_id, "prompt")
            results.append(result)
            return result

    # Run 5 evals with max 2 concurrent
    await asyncio.gather(*[
        bounded_eval(f"tc_{i:03d}") for i in range(5)
    ])

    assert len(results) == 5
    assert all(isinstance(r, EvalResult) for r in results)
