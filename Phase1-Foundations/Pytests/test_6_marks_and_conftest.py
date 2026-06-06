# tests/test_6_marks_and_conftest.py
# ---------------------------------------------------------------
# TOPIC: Marks + conftest.py pattern
#
# What you'll learn:
#   - skip, skipif, xfail marks
#   - What conftest.py is and how it works
#   - Custom marks for grouping tests
#
# Run: pytest tests/test_6_marks_and_conftest.py -v
# Run only "slow" marked tests: pytest -m slow -v
# Run everything except "slow": pytest -m "not slow" -v
# ---------------------------------------------------------------

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import EvalResult, EvalScore, is_passing


# ---------------------------------------------------------------
# PART A — skip
# Skip a test unconditionally.
# Use when a feature isn't built yet.
# ---------------------------------------------------------------

@pytest.mark.skip(reason="Dashboard not built yet")
def test_dashboard_renders():
    # This test won't run at all
    # Remove the mark when you implement the feature
    assert False, "This should never run"


@pytest.mark.skip(reason="Requires real PostgreSQL connection")
def test_save_run_to_db():
    pass


# ---------------------------------------------------------------
# PART B — skipif
# Skip based on a condition.
# Use when a test only makes sense in certain environments.
# ---------------------------------------------------------------

import platform

@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="File path handling differs on Windows"
)
def test_unix_file_path():
    path = "/home/claude/evalforge/test_cases/async_await.yaml"
    assert path.startswith("/")


@pytest.mark.skipif(
    os.environ.get("ANTHROPIC_API_KEY") is None,
    reason="No API key set — skipping real API test"
)
def test_real_api_call():
    # Only runs if ANTHROPIC_API_KEY is in environment
    # Safe to have in the test suite — just skips in local dev
    pass


# ---------------------------------------------------------------
# PART C — xfail (expected failure)
# Mark a test you EXPECT to fail — a known bug or unfinished feature.
# xfail tests don't count as failures in CI.
# Remove the mark once the bug is fixed.
# ---------------------------------------------------------------

@pytest.mark.xfail(reason="Known bug: coercion of None score not handled yet")
def test_none_score_handling():
    # This is expected to fail — Pydantic raises an error
    # Once you add None handling in the validator, this will pass
    # Then remove the xfail mark
    score = EvalScore(score=None)
    assert score.score == 0.0


@pytest.mark.xfail(strict=True, reason="This MUST fail — if it passes something is wrong")
def test_impossible_condition():
    # strict=True means: if this unexpectedly PASSES, count it as a failure
    # Use when you're certain something should always fail
    assert 1 == 2


# ---------------------------------------------------------------
# PART D — Custom marks for grouping
# Define your own marks to group and filter tests.
# Register them in pytest.ini to avoid warnings:
#
# [pytest]
# markers =
#     slow: marks tests as slow
#     integration: marks tests that need external services
#     unit: marks pure unit tests
# ---------------------------------------------------------------

@pytest.mark.slow
def test_large_eval_batch():
    """Marked slow — excluded from quick dev runs."""
    results = [
        EvalResult(f"tc_{i:03d}", 4.2, True)
        for i in range(1000)
    ]
    assert len(results) == 1000


@pytest.mark.slow
def test_another_slow_operation():
    import time
    time.sleep(0.1)     # simulate slow work
    assert True


@pytest.mark.integration
def test_db_integration():
    """Marked integration — only runs in CI with real services."""
    # Would connect to real PostgreSQL in CI
    pass


@pytest.mark.unit
def test_is_passing_unit():
    """Marked unit — fast, no dependencies."""
    assert is_passing(4.2) is True


# ---------------------------------------------------------------
# PART E — conftest.py explained
#
# conftest.py is a special file pytest finds automatically.
# Put fixtures there that multiple test files need.
# No import required — pytest loads it before running tests.
#
# File structure:
#   tests/
#   ├── conftest.py          ← loaded automatically
#   ├── test_1_basics.py     ← can use fixtures from conftest
#   ├── test_2_fixtures.py   ← same
#   └── test_5_mocking.py    ← same
#
# What to put in conftest.py:
#   - Session-scoped fixtures (DB connections, loaded configs)
#   - Shared test data used across many files
#   - Common async fixtures
# ---------------------------------------------------------------

# The conftest.py for this project would look like:
#
# # tests/conftest.py
# import pytest
# from models import EvalResult, EvalRun
#
# @pytest.fixture(scope="session")
# def shared_passing_result():
#     return EvalResult("tc_shared", 4.2, True)
#
# @pytest.fixture(scope="session")
# def shared_model_config():
#     return {"model": "claude-sonnet-4-6", "max_tokens": 1000}
#
# Then in any test file — no import needed:
#
# def test_something(shared_passing_result):
#     assert shared_passing_result.score == 4.2

# Test that uses a fixture defined above (not conftest, just local)
def test_custom_mark_with_fixture():
    result = EvalResult("tc_mark", 3.5, True)
    assert result.passed is True
