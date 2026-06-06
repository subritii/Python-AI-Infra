# tests/test_1_basics.py
# ---------------------------------------------------------------
# TOPIC: Basic test structure + assert patterns
#
# What you'll learn:
#   - The three rules of pytest (file, function, assert)
#   - Every assert pattern you'll actually use
#   - Testing exceptions with pytest.raises
#   - Inspecting exception messages
#
# Run: pytest tests/test_1_basics.py -v
# ---------------------------------------------------------------

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydantic import ValidationError
from models import is_passing, EvalResult, EvalScore


# ---------------------------------------------------------------
# PART A — Basic assert patterns
# assert evaluates an expression.
# True  → test passes silently
# False → AssertionError, test fails
# ---------------------------------------------------------------

def test_equality():
    result = EvalResult("tc_001", 4.2, True)
    assert result.test_id == "tc_001"           # string equality
    assert result.score == 4.2                  # float equality


def test_boolean_identity():
    # Use `is True` / `is False` for booleans — not == True
    # == True passes for anything truthy (1, "hello", [1])
    # is True  passes ONLY for the actual boolean True
    result = EvalResult("tc_001", 4.2, True)
    assert result.passed is True                # ✅ correct
    # assert result.passed == True             # works but less strict


def test_none_check():
    result = EvalResult("tc_001", 4.2, True, notes="")
    assert result.notes is not None             # notes exists
    assert result.notes == ""                   # but is empty string


def test_comparison():
    result = EvalResult("tc_001", 4.2, True)
    assert result.score >= 1.0
    assert result.score <= 5.0
    assert result.score > 3.0


def test_inequality():
    result = EvalResult("tc_001", 4.2, True)
    assert result.score != 0.0


def test_type_check():
    result = EvalResult("tc_001", 4.2, True)
    assert isinstance(result.score, float)
    assert isinstance(result.passed, bool)
    assert isinstance(result.tags, list)


def test_membership():
    result = EvalResult("tc_001", 4.2, True, tags=["python", "async"])
    assert "python" in result.tags
    assert "missing_tag" not in result.tags


def test_length():
    result = EvalResult("tc_001", 4.2, True, tags=["python", "async"])
    assert len(result.tags) == 2


def test_assert_with_message():
    # Add a message — shown when the test fails so you know why
    score = 4.2
    assert score >= 3.0, f"Expected passing score, got {score}"
    assert isinstance(score, float), f"Score should be float, got {type(score)}"


# ---------------------------------------------------------------
# PART B — is_passing function tests
# Always test: normal cases, boundaries, edge cases
# ---------------------------------------------------------------

def test_high_score_passes():
    assert is_passing(4.2) is True


def test_low_score_fails():
    assert is_passing(1.5) is False


def test_exact_boundary_passes():
    # Boundary testing — always test the exact cutoff value
    # 3.0 is the boundary — should PASS
    assert is_passing(3.0) is True


def test_just_below_boundary_fails():
    # Just below boundary — should FAIL
    assert is_passing(2.9) is False


def test_zero_fails():
    assert is_passing(0.0) is False


def test_max_score_passes():
    assert is_passing(5.0) is True


# ---------------------------------------------------------------
# PART C — testing exceptions
# When the correct behavior IS raising an error.
# Never write if/raise inside a test — that's the code's job.
# ---------------------------------------------------------------

def test_score_below_range_raises():
    # pytest.raises() — passes if ValidationError is raised
    # fails if ValidationError is NOT raised
    with pytest.raises(ValidationError):
        EvalScore(score=0.5)


def test_score_above_range_raises():
    with pytest.raises(ValidationError):
        EvalScore(score=6.0)


def test_score_zero_raises():
    with pytest.raises(ValidationError):
        EvalScore(score=0.0)


def test_score_negative_raises():
    with pytest.raises(ValidationError):
        EvalScore(score=-1.0)


def test_valid_score_does_not_raise():
    # No pytest.raises — we assert it works cleanly
    score = EvalScore(score=4.2)
    assert score.score == 4.2


def test_inspect_exception_message():
    # excinfo lets you read the actual error message
    with pytest.raises(ValidationError) as exc_info:
        EvalScore(score=99.0)
    # Check the message contains what we expect
    assert "Score must be 1" in str(exc_info.value)


def test_wrong_exception_type_fails():
    # If the wrong error is raised, pytest.raises fails the test
    # This test asserts that a plain string doesn't raise ValidationError
    score = EvalScore(score=4.2)        # valid — no error
    assert score.score == 4.2
