# phase3/01_anthropic_sdk/02_async_client.py
# -------------------------------------------------------
# TOPIC: Async client + concurrent API calls
#
# What you'll learn:
#   - Difference between sync and async client
#   - Running multiple calls concurrently with gather
#   - Semaphore for rate limiting concurrent calls
#   - How concurrency speeds up batch eval runs
#
# Run: python 02_async_client.py
# -------------------------------------------------------

import asyncio
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


# -------------------------------------------------------
# Mock async call — simulates network latency
# -------------------------------------------------------

async def mock_async_call(prompt: str, call_id: int = 0) -> APIResponse:
    """Simulates an async API call with 0.5s latency."""
    await asyncio.sleep(0.5)    # simulate network round trip
    print(f"  [call {call_id}] completed")
    return APIResponse(
        text=f"Mock response for: {prompt[:40]}",
        input_tokens=len(prompt) // 4,
        output_tokens=20,
        cost_usd=0.0
    )


# -------------------------------------------------------
# Real async call
# -------------------------------------------------------

async def real_async_call(prompt: str, client) -> APIResponse:
    import anthropic
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return APIResponse(
        text=response.content[0].text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=(response.usage.input_tokens * 0.000003) +
                 (response.usage.output_tokens * 0.000015)
    )


# -------------------------------------------------------
# DEMO 1 — Sequential vs concurrent
# Shows why async matters for batch evals
# -------------------------------------------------------

async def demo_sequential_vs_concurrent():
    print("\n--- DEMO 1: Sequential vs Concurrent ---")
    prompts = [
        "What is async/await?",
        "What is a token?",
        "What is RAG?",
        "What is temperature in LLMs?",
        "What is a context window?"
    ]

    # Sequential — each call waits for the previous
    print("\nSequential (one at a time):")
    start = time.perf_counter()
    results_seq = []
    for i, prompt in enumerate(prompts):
        result = await mock_async_call(prompt, call_id=i)
        results_seq.append(result)
    seq_time = time.perf_counter() - start
    print(f"  Time: {seq_time:.2f}s  (5 calls × 0.5s each)")

    # Concurrent — all calls fire at once
    print("\nConcurrent (all at once with gather):")
    start = time.perf_counter()
    results_con = await asyncio.gather(*[
        mock_async_call(prompt, call_id=i)
        for i, prompt in enumerate(prompts)
    ])
    con_time = time.perf_counter() - start
    print(f"  Time: {con_time:.2f}s  (5 calls run simultaneously)")
    print(f"\n  Speedup: {seq_time / con_time:.1f}x faster with concurrency")


# -------------------------------------------------------
# DEMO 2 — Semaphore rate limiting
# Real Anthropic API has rate limits — you need to control
# how many calls fire at once
# -------------------------------------------------------

async def demo_semaphore():
    print("\n--- DEMO 2: Semaphore Rate Limiting ---")
    print("Limits concurrent calls to avoid hitting API rate limits")

    prompts = [f"Question {i}" for i in range(10)]
    sem = asyncio.Semaphore(3)      # max 3 concurrent calls at once
    results = []

    async def bounded_call(prompt: str, call_id: int):
        async with sem:             # blocks if 3 are already running
            print(f"  [call {call_id}] started  (sem slots used: blocking others if > 3)")
            result = await mock_async_call(prompt, call_id)
            results.append(result)
            return result

    start = time.perf_counter()
    await asyncio.gather(*[
        bounded_call(prompt, i)
        for i, prompt in enumerate(prompts)
    ])
    elapsed = time.perf_counter() - start

    print(f"\n  10 calls, max 3 concurrent → {elapsed:.2f}s")
    print(f"  Without semaphore: ~0.5s (all fire at once — risks rate limit)")
    print(f"  With semaphore(3): ~{elapsed:.1f}s (controlled — safe)")


# -------------------------------------------------------
# DEMO 3 — Collecting results with cost tracking
# This is the eval runner pattern
# -------------------------------------------------------

async def demo_batch_with_cost():
    print("\n--- DEMO 3: Batch Eval Run with Cost Tracking ---")

    test_cases = [
        ("tc_001", "What is async def?"),
        ("tc_002", "What is a Semaphore?"),
        ("tc_003", "What is top-p sampling?"),
    ]

    sem = asyncio.Semaphore(2)
    total_cost = 0.0

    async def run_one(test_id: str, prompt: str) -> dict:
        async with sem:
            result = await mock_async_call(prompt, call_id=test_id)
            return {
                "test_id": test_id,
                "response": result.text,
                "tokens": result.input_tokens + result.output_tokens,
                "cost": result.cost_usd
            }

    results = await asyncio.gather(*[
        run_one(tid, prompt)
        for tid, prompt in test_cases
    ])

    for r in results:
        total_cost += r["cost"]
        print(f"  {r['test_id']} | tokens: {r['tokens']} | cost: ${r['cost']:.6f}")

    print(f"\n  Total run cost: ${total_cost:.6f}")
    print(f"  Results: {len(results)} test cases evaluated")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 50)
    print("ASYNC CLIENT")
    print("=" * 50)
    print(f"MOCK_MODE: {MOCK_MODE}")

    await demo_sequential_vs_concurrent()
    await demo_semaphore()
    await demo_batch_with_cost()

    print("\n" + "=" * 50)
    print("Key takeaways:")
    print("  1. Use AsyncAnthropic() — not Anthropic() — for batch runs")
    print("  2. asyncio.gather() runs calls concurrently — much faster")
    print("  3. Semaphore controls concurrency — prevents rate limit errors")
    print("  4. Track cost from response.usage — not estimated")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
