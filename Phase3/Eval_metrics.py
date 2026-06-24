# TOPIC: Eval metrics — exact match, BLEU, semantic similarity
#
# What you'll learn:
#   - Exact match and fuzzy matching
#   - BLEU score (word overlap)
#   - Semantic similarity (meaning comparison)
#   - The routing logic that picks the right metric per test case
#   - How all metrics output on the same 1–5 scale
#
# Install: pip install nltk numpy
# -------------------------------------------------------

import re
import json
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# nltk needed for BLEU — download data on first run
import nltk
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# -------------------------------------------------------
# TestCase — input shape
# The `metric` field tells the scorer which method to use
# -------------------------------------------------------

@dataclass
class TestCase:
    id: str
    topic: str
    prompt: str
    expected_output: str
    metric: str = "llm_judge"    # "exact", "fuzzy", "bleu", "semantic", "llm_judge"
    difficulty: int = 1
    tags: list = field(default_factory=list)


@dataclass
class MetricResult:
    test_id: str
    metric_used: str
    raw_value: float       # raw metric output (0.0–1.0 or True/False)
    score: float           # normalized to 1.0–5.0
    passed: bool           # True if score >= 3.0
    model_output: str
    expected: str


# -------------------------------------------------------
# METRIC 1 — Exact match
# Strictest — character for character equality
# -------------------------------------------------------

def exact_match(output: str, expected: str) -> bool:
    """
    Returns True only if output equals expected after
    stripping whitespace and lowercasing.
    """
    return output.strip().lower() == expected.strip().lower()


# -------------------------------------------------------
# METRIC 2 — Fuzzy / normalized match
# Looser — strips punctuation and extra spaces
# -------------------------------------------------------

def normalize(text: str) -> str:
    """Remove punctuation, collapse whitespace, lowercase."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)   # remove all punctuation
    text = re.sub(r'\s+', ' ', text)      # collapse multiple spaces
    return text


def fuzzy_match(output: str, expected: str) -> bool:
    """True if output matches expected after normalization."""
    return normalize(output) == normalize(expected)


def contains_match(output: str, expected: str) -> bool:
    """True if expected appears anywhere inside output."""
    return normalize(expected) in normalize(output)


# -------------------------------------------------------
# METRIC 3 — BLEU score
# Measures word sequence overlap (n-gram overlap)
# -------------------------------------------------------

def bleu_score(output: str, expected: str) -> float:
    """
    Returns 0.0 (no overlap) to 1.0 (perfect overlap).
    Measures how many word sequences in output appear in expected.
    """
    reference  = [expected.lower().split()]   # must be list of lists
    hypothesis = output.lower().split()
    smoothing  = SmoothingFunction().method1  # handles short texts gracefully
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)


# -------------------------------------------------------
# METRIC 4 — Semantic similarity
# Measures meaning closeness using embeddings + cosine similarity
# -------------------------------------------------------

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Measures angle between two vectors.
    1.0 = same direction = same meaning
    0.0 = perpendicular = unrelated
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def mock_embedding(text: str) -> list:
    """
    Fake embedding — creates a vector from character codes.
    Real usage: call OpenAI/Anthropic embedding API.
    Shape is the same — a list of floats.
    """
    vec = np.zeros(64)
    for i, char in enumerate(text.lower()[:64]):
        vec[i] = ord(char) / 128.0
    return vec.tolist()


async def semantic_similarity(output: str, expected: str) -> float:
    """
    Returns 0.0 to 1.0.
    In real usage: replace mock_embedding with actual API call.
    """
    output_vec   = mock_embedding(output)
    expected_vec = mock_embedding(expected)
    return cosine_similarity(output_vec, expected_vec)


# -------------------------------------------------------
# Score normalizer — maps all metrics to 1.0–5.0 scale
# -------------------------------------------------------

def normalize_to_score(raw_value: float, passed: bool = None) -> tuple[float, bool]:
    """
    Converts raw metric output to the 1.0–5.0 scale used across EvalForge.
    Returns (score, passed).
    """
    if passed is not None:
        # Boolean input (exact match) — binary scoring
        score = 5.0 if passed else 1.0
        return score, passed

    # Float input (BLEU, semantic) — scale 0.0–1.0 to 1.0–5.0
    score = round(1.0 + (raw_value * 4.0), 1)
    score = max(1.0, min(5.0, score))    # clamp to valid range
    return score, score >= 3.0


# -------------------------------------------------------
# Mock LLM judge — returns a fixed score
# In Phase 4 this is replaced by judge_output()
# -------------------------------------------------------

async def mock_llm_judge(test_case: TestCase, model_output: str) -> MetricResult:
    return MetricResult(
        test_id=test_case.id,
        metric_used="llm_judge",
        raw_value=0.84,
        score=4.2,
        passed=True,
        model_output=model_output,
        expected=test_case.expected_output
    )


# -------------------------------------------------------
# The router — picks the right metric per test case
# This is the key function — all metrics flow through here
# -------------------------------------------------------

async def score_output(test_case: TestCase, model_output: str) -> MetricResult:
    """
    Routes to the right metric based on test_case.metric.
    All paths return a MetricResult with score on 1.0–5.0 scale.
    """

    if test_case.metric == "exact":
        passed = exact_match(model_output, test_case.expected_output)
        score, passed = normalize_to_score(None, passed)
        return MetricResult(
            test_id=test_case.id, metric_used="exact",
            raw_value=1.0 if passed else 0.0, score=score, passed=passed,
            model_output=model_output, expected=test_case.expected_output
        )

    elif test_case.metric == "fuzzy":
        passed = fuzzy_match(model_output, test_case.expected_output)
        score, passed = normalize_to_score(None, passed)
        return MetricResult(
            test_id=test_case.id, metric_used="fuzzy",
            raw_value=1.0 if passed else 0.0, score=score, passed=passed,
            model_output=model_output, expected=test_case.expected_output
        )

    elif test_case.metric == "bleu":
        raw   = bleu_score(model_output, test_case.expected_output)
        score, passed = normalize_to_score(raw)
        return MetricResult(
            test_id=test_case.id, metric_used="bleu",
            raw_value=round(raw, 3), score=score, passed=passed,
            model_output=model_output, expected=test_case.expected_output
        )

    elif test_case.metric == "semantic":
        raw   = await semantic_similarity(model_output, test_case.expected_output)
        score, passed = normalize_to_score(raw)
        return MetricResult(
            test_id=test_case.id, metric_used="semantic",
            raw_value=round(raw, 3), score=score, passed=passed,
            model_output=model_output, expected=test_case.expected_output
        )

    else:  # llm_judge
        return await mock_llm_judge(test_case, model_output)


# -------------------------------------------------------
# DEMO 1 — Each metric on its own
# -------------------------------------------------------

def demo_individual_metrics():
    print("\n--- DEMO 1: Each Metric Individually ---\n")

    expected = "async def marks a coroutine function"

    outputs = [
        ("Perfect match",   "async def marks a coroutine function"),
        ("Extra words",     "async def marks a coroutine function that can be awaited"),
        ("Same meaning",    "async makes a function into a coroutine"),
        ("Partial correct", "async is used for slow network operations"),
        ("Wrong answer",    "async runs threads in parallel"),
    ]

    print(f"  Expected: '{expected}'\n")
    print(f"  {'Output':<45} {'Exact':<7} {'Fuzzy':<7} {'BLEU':<7} {'Sem'}")
    print("  " + "-" * 75)

    for label, output in outputs:
        em   = "✅" if exact_match(output, expected) else "❌"
        fm   = "✅" if fuzzy_match(output, expected) else "❌"
        bl   = round(bleu_score(output, expected), 2)
        sem  = round(cosine_similarity(mock_embedding(output), mock_embedding(expected)), 2)
        print(f"  {label:<45} {em:<7} {fm:<7} {bl:<7} {sem}")


# -------------------------------------------------------
# DEMO 2 — The routing logic
# -------------------------------------------------------

async def demo_routing():
    print("\n--- DEMO 2: Routing Logic ---")
    print("Different test cases use different metrics\n")

    test_cases = [
        TestCase("tc_001", "capitals",     "What is the capital of France?",     "Paris",                              metric="exact"),
        TestCase("tc_002", "sentiment",    "Classify: 'I love this product'",    "positive",                           metric="exact"),
        TestCase("tc_003", "async/await",  "What does async def do?",            "async def marks a coroutine function", metric="semantic"),
        TestCase("tc_004", "summarize",    "Summarize this paragraph",           "The topic covers key AI concepts",   metric="bleu"),
        TestCase("tc_005", "code review",  "Is there a bug in this code?",       "Yes — missing await keyword",        metric="llm_judge"),
    ]

    model_outputs = {
        "tc_001": "Paris",
        "tc_002": "positive",
        "tc_003": "async def makes a function a coroutine that can be awaited",
        "tc_004": "The paragraph discusses important AI concepts and methods",
        "tc_005": "Yes, the await keyword is missing on line 3",
    }

    print(f"  {'ID':<8} {'Metric':<12} {'Raw':<7} {'Score':<7} {'Pass'}")
    print("  " + "-" * 50)

    for tc in test_cases:
        output = model_outputs[tc.id]
        result = await score_output(tc, output)
        icon   = "✅" if result.passed else "❌"
        print(f"  {tc.id:<8} {result.metric_used:<12} {result.raw_value:<7} {result.score:<7} {icon}")


# -------------------------------------------------------
# DEMO 3 — All on same 1–5 scale
# -------------------------------------------------------

async def demo_normalized_scale():
    print("\n--- DEMO 3: All Metrics on the Same 1–5 Scale ---")
    print("Different metrics, comparable scores\n")

    expected = "async def marks a coroutine function"
    output   = "async def makes a function into a coroutine"

    bleu_raw = bleu_score(output, expected)
    sem_raw  = cosine_similarity(mock_embedding(output), mock_embedding(expected))

    bleu_score_norm, bleu_passed = normalize_to_score(bleu_raw)
    sem_score_norm,  sem_passed  = normalize_to_score(sem_raw)

    print(f"  Output:   '{output}'")
    print(f"  Expected: '{expected}'\n")
    print(f"  Exact match:          {'pass' if exact_match(output, expected) else 'fail'} → score: {'5.0' if exact_match(output, expected) else '1.0'}")
    print(f"  BLEU raw:  {bleu_raw:.3f}  → score: {bleu_score_norm}")
    print(f"  Semantic:  {sem_raw:.3f}  → score: {sem_score_norm}")
    print(f"  LLM judge: (mocked)   → score: 4.2")
    print(f"\n  All on 1–5 scale → pass_rate and avg_score are meaningful across metric types")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 55)
    print("EVAL METRICS")
    print("=" * 55)

    demo_individual_metrics()
    await demo_routing()
    await demo_normalized_scale()

    print("\n" + "=" * 55)
    print("Key takeaways:")
    print("  1. Exact match — free, instant, only for exact answers")
    print("  2. BLEU — word overlap, good for summaries")
    print("  3. Semantic — meaning closeness, best for flexible outputs")
    print("  4. Router picks cheapest metric that works for each case")
    print("  5. Always normalize to 1–5 so scores are comparable")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())