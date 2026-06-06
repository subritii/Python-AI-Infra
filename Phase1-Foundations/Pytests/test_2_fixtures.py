# tests/test_2_fixtures.py
# ---------------------------------------------------------------
# TOPIC: Fixtures
#
# What you'll learn:
#   - Defining and injecting fixtures
#   - Fixtures using other fixtures
#   - Fixture scope (function vs session)
#   - yield fixtures for setup + teardown
#
# Run: pytest tests/test_2_fixtures.py -v
# Run with print output: pytest tests/test_2_fixtures.py -v -s
# ---------------------------------------------------------------

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import EvalResult, EvalRun, EvalScore, compute_run_stats


# ---------------------------------------------------------------
# PART A — Basic fixtures
# Define once with @pytest.fixture
# Inject by adding as a parameter — name must match exactly
# ---------------------------------------------------------------

@pytest.fixture
def passing_result():
    """A standard passing eval result."""
    return EvalResult(test_id="tc_001", score=4.2, passed=True)


@pytest.fixture
def failing_result():
    """A failing eval result — score below threshold."""
    return EvalResult(test_id="tc_002", score=1.5, passed=False)


@pytest.fixture
def boundary_result():
    """An eval result at exactly the passing boundary."""
    return EvalResult(test_id="tc_003", score=3.0, passed=True)


def test_passing_score(passing_result):             # pytest injects by name
    assert passing_result.score == 4.2


def test_passing_is_true(passing_result):           # same fixture, fresh each time
    assert passing_result.passed is True


def test_failing_score(failing_result):
    assert failing_result.score == 1.5


def test_failing_is_false(failing_result):
    assert failing_result.passed is False


def test_boundary_passes(boundary_result):
    assert boundary_result.passed is True


def test_two_fixtures(passing_result, failing_result):
    # Use multiple fixtures in one test
    assert passing_result.score > failing_result.score


# ---------------------------------------------------------------
# PART B — Fixtures using other fixtures
# Fixtures can take other fixtures as parameters
# ---------------------------------------------------------------

@pytest.fixture
def empty_run():
    """A fresh eval run with no results."""
    return EvalRun(run_id="run_001", model_version="claude-sonnet-4-6")


@pytest.fixture
def populated_run(passing_result, failing_result):
    """A run with results — uses the fixtures defined above."""
    run = EvalRun(run_id="run_002", model_version="claude-sonnet-4-6")
    run.results = [passing_result, failing_result]
    run.total_cost = 0.004
    return run


def test_empty_run_has_no_results(empty_run):
    assert len(empty_run.results) == 0
    assert empty_run.total_cost == 0.0


def test_populated_run_count(populated_run):
    assert len(populated_run.results) == 2


def test_populated_run_cost(populated_run):
    assert populated_run.total_cost == 0.004


def test_run_stats_empty(empty_run):
    stats = compute_run_stats(empty_run)
    assert stats["total"] == 0
    assert stats["pass_rate"] == 0.0


def test_run_stats_populated(populated_run):
    stats = compute_run_stats(populated_run)
    assert stats["total"] == 2
    assert stats["pass_rate"] == 0.5              # 1 pass, 1 fail = 50%
    assert stats["avg_score"] == 2.85             # (4.2 + 1.5) / 2


# ---------------------------------------------------------------
# PART C — Fixture scope
# function (default) — new fixture for every test
# session           — one fixture for entire test run
# ---------------------------------------------------------------

@pytest.fixture(scope="function")
def function_scoped():
    """Created fresh for every test. Default behavior."""
    print("\n  [function_scoped] created")
    return EvalResult("tc_func", 3.5, True)


@pytest.fixture(scope="session")
def session_scoped():
    """Created ONCE for the entire test session.
    Use for expensive setup — DB connections, loaded configs."""
    print("\n  [session_scoped] created — only happens once")
    return {"model": "claude-sonnet-4-6", "max_tokens": 1000}


def test_function_scope_1(function_scoped):
    # function_scoped was freshly created for this test
    assert function_scoped.score == 3.5


def test_function_scope_2(function_scoped):
    # function_scoped was freshly created AGAIN for this test
    assert function_scoped.passed is True


def test_session_scope_1(session_scoped):
    # session_scoped was created once — same object reused
    assert session_scoped["model"] == "claude-sonnet-4-6"


def test_session_scope_2(session_scoped):
    # Same session_scoped object — NOT recreated
    assert session_scoped["max_tokens"] == 1000


# ---------------------------------------------------------------
# PART D — yield fixtures (setup + teardown)
# yield = run test here, then clean up after
# Everything before yield  = setup
# Everything after yield   = teardown (runs even if test fails)
# ---------------------------------------------------------------

@pytest.fixture
def result_with_cleanup():
    # --- SETUP ---
    print("\n  [setup] creating result")
    result = EvalResult("tc_cleanup", 4.0, True)

    yield result        # ← test runs here, receives the result

    # --- TEARDOWN --- runs after test, even on failure
    print("\n  [teardown] cleaning up result")
    # In real code: close DB connection, delete temp files, etc.


@pytest.fixture(scope="session")
def db_connection_simulation():
    """Simulates an expensive session-scoped resource."""
    print("\n  [session setup] opening DB connection")
    connection = {"status": "open", "queries": 0}

    yield connection

    print("\n  [session teardown] closing DB connection")
    connection["status"] = "closed"


def test_with_cleanup(result_with_cleanup):
    # Run pytest -s to see the setup/teardown print statements
    assert result_with_cleanup.test_id == "tc_cleanup"
    assert result_with_cleanup.score == 4.0


def test_db_connection_1(db_connection_simulation):
    assert db_connection_simulation["status"] == "open"


def test_db_connection_2(db_connection_simulation):
    # Same connection object — not recreated
    assert db_connection_simulation["status"] == "open"
