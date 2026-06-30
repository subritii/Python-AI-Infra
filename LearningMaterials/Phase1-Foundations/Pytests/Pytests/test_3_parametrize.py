# tests/test_3_parametrize.py
# ---------------------------------------------------------------
# TOPIC: Parametrize
#
# What you'll learn:
#   - Running one test function with many inputs
#   - Parametrizing with single and multiple values
#   - Combining parametrize with fixtures
#   - Naming parametrized cases for readable output
#
# Run: pytest tests/test_3_parametrize.py -v
# ---------------------------------------------------------------

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydantic import ValidationError
from models import is_passing, EvalResult, EvalScore


# ---------------------------------------------------------------
# PART A — The problem without parametrize
# Without it you write the same test 5 times with different values
# ---------------------------------------------------------------

# BAD — repetitive, hard to maintain
def test_score_5_passes():
    assert is_passing(5.0) is True

def test_score_3_passes():
    assert is_passing(3.0) is True

def test_score_2_fails():
    assert is_passing(2.9) is False

# There's a better way ↓


# ---------------------------------------------------------------
# PART B — Single parameter
# @pytest.mark.parametrize("param_name", [value1, value2, ...])
# pytest runs the test once per value
# ---------------------------------------------------------------

@pytest.mark.parametrize("bad_score", [0.0, 0.5, 5.1, 99.0, -1.0])
def test_invalid_scores_raise(bad_score):
    """Runs 5 times — once for each bad score value."""
    with pytest.raises(ValidationError):
        EvalScore(score=bad_score)


@pytest.mark.parametrize("good_score", [1.0, 2.5, 3.0, 4.2, 5.0])
def test_valid_scores_accepted(good_score):
    """Runs 5 times — once for each valid score value."""
    s = EvalScore(score=good_score)
    assert s.score == good_score


# ---------------------------------------------------------------
# PART C — Multiple parameters
# @pytest.mark.parametrize("a, b", [(val_a, val_b), ...])
# Each tuple = one test run
# ---------------------------------------------------------------

@pytest.mark.parametrize("score, expected", [
    (5.0,  True),    # max score — passes
    (3.0,  True),    # exact boundary — passes
    (2.9,  False),   # just below boundary — fails
    (1.5,  False),   # low score — fails
    (0.0,  False),   # zero — fails
])
def test_is_passing_parametrized(score, expected):
    """Runs 5 times. Each row is (score, expected_result)."""
    assert is_passing(score) is expected


@pytest.mark.parametrize("test_id, score, should_pass", [
    ("tc_001", 4.2,  True),
    ("tc_002", 1.5,  False),
    ("tc_003", 3.0,  True),
    ("tc_004", 2.99, False),
])
def test_eval_result_pass_status(test_id, score, should_pass):
    result = EvalResult(test_id=test_id, score=score, passed=should_pass)
    assert result.passed is should_pass
    assert result.score == score


# ---------------------------------------------------------------
# PART D — Named test cases
# Add an id to each case for readable output in pytest -v
# ---------------------------------------------------------------

@pytest.mark.parametrize("score, expected", [
    pytest.param(5.0, True,  id="max_score"),
    pytest.param(3.0, True,  id="exact_boundary"),
    pytest.param(2.9, False, id="just_below_boundary"),
    pytest.param(0.0, False, id="zero_score"),
], )
def test_named_cases(score, expected):
    """In pytest -v output, you'll see test_named_cases[max_score] etc.
    Much more readable than test_named_cases[5.0-True]."""
    assert is_passing(score) is expected


# ---------------------------------------------------------------
# PART E — Combining parametrize with fixtures
# ---------------------------------------------------------------

@pytest.fixture
def base_run_id():
    return "run_parametrized"


@pytest.mark.parametrize("score", [1.0, 2.5, 4.0, 5.0])
def test_eval_result_score_stored(score, base_run_id):
    """Parametrize + fixture together — 4 test runs, each with the fixture."""
    result = EvalResult(test_id=f"{base_run_id}_{score}", score=score, passed=score >= 3.0)
    assert result.score == score
    assert result.test_id.startswith(base_run_id)


# ---------------------------------------------------------------
# PART F — Stacked parametrize (cartesian product)
# Two @parametrize decorators = all combinations
# 3 scores × 2 tags = 6 test runs
# ---------------------------------------------------------------

@pytest.mark.parametrize("score", [1.5, 3.0, 4.2])
@pytest.mark.parametrize("tag", ["python", "async"])
def test_result_with_tags(score, tag):
    """Runs 6 times: every combination of score and tag."""
    result = EvalResult("tc_stack", score, score >= 3.0, tags=[tag])
    assert tag in result.tags
    assert result.score == score
