# phase3/01_anthropic_sdk/04_error_handling.py
# -------------------------------------------------------
# TOPIC: Error handling + exponential backoff
#
# What you'll learn:
#   - The 4 Anthropic error types and when each fires
#   - Exponential backoff for rate limits
#   - Retry logic with informed error messages
#   - How to handle errors in a batch eval run
#
# Run: python 04_error_handling.py
# -------------------------------------------------------

import asyncio
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------
# Simulated error classes — same names as real Anthropic errors
# -------------------------------------------------------

class RateLimitError(Exception):
    """Too many requests — hit token or request per minute limit."""
    pass

class APIStatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

class APIConnectionError(Exception):
    """Network issue — no response received."""
    pass

class AuthenticationError(Exception):
    """Invalid API key."""
    pass


# -------------------------------------------------------
# Mock API that fails on purpose — for learning
# -------------------------------------------------------

call_count = 0

async def flaky_api_call(prompt: str, fail_mode: str = "none") -> str:
    """
    Simulates API calls that fail in different ways.
    fail_mode options: "rate_limit", "server_error", "connection", "auth", "none"
    """
    global call_count
    call_count += 1
    await asyncio.sleep(0.1)   # simulate latency

    if fail_mode == "rate_limit":
        raise RateLimitError("Rate limit exceeded: 100k tokens/min")

    if fail_mode == "server_error":
        raise APIStatusError("Internal server error", status_code=500)

    if fail_mode == "client_error":
        raise APIStatusError("Invalid model specified", status_code=400)

    if fail_mode == "connection":
        raise APIConnectionError("Connection timed out")

    if fail_mode == "auth":
        raise AuthenticationError("Invalid API key")

    if fail_mode == "flaky":
        # Fails first 2 times, succeeds on 3rd — simulates transient error
        if call_count <= 2:
            raise RateLimitError(f"Rate limited (attempt {call_count})")

    return f"Success: {prompt[:50]}"


# -------------------------------------------------------
# DEMO 1 — The 4 error types and how to handle each
# -------------------------------------------------------

async def demo_error_types():
    print("\n--- DEMO 1: The 4 Error Types ---")

    errors = [
        ("rate_limit",   "RateLimitError   — too many requests, wait and retry"),
        ("server_error", "APIStatusError 5xx — server problem, retry with backoff"),
        ("client_error", "APIStatusError 4xx — your mistake, don't retry"),
        ("connection",   "APIConnectionError — network issue, retry"),
        ("auth",         "AuthenticationError — bad API key, don't retry"),
    ]

    for fail_mode, description in errors:
        try:
            await flaky_api_call("test prompt", fail_mode=fail_mode)
        except RateLimitError as e:
            print(f"  ⚠️  {description}")
            print(f"      Action: wait 60s and retry")
        except APIStatusError as e:
            if e.status_code >= 500:
                print(f"  ⚠️  {description}")
                print(f"      Action: retry with exponential backoff")
            else:
                print(f"  ❌ {description}")
                print(f"      Action: fix the request — do not retry")
        except APIConnectionError as e:
            print(f"  ⚠️  {description}")
            print(f"      Action: check network, retry with backoff")
        except AuthenticationError as e:
            print(f"  🔑 {description}")
            print(f"      Action: check .env file — do not retry")


# -------------------------------------------------------
# DEMO 2 — Exponential backoff
# Retries with increasing wait: 1s, 2s, 4s, 8s, 16s
# -------------------------------------------------------

async def call_with_backoff(
    prompt: str,
    fail_mode: str = "none",
    max_retries: int = 5
) -> str:
    """
    Retries failed calls with exponential backoff.
    Only retries on transient errors (rate limits, server errors).
    Immediately raises on permanent errors (auth, bad request).
    """
    global call_count
    call_count = 0   # reset for demo

    for attempt in range(max_retries):
        try:
            result = await flaky_api_call(prompt, fail_mode=fail_mode)
            if attempt > 0:
                print(f"    ✅ Succeeded on attempt {attempt + 1}")
            return result

        except AuthenticationError:
            raise   # permanent — don't retry

        except APIStatusError as e:
            if e.status_code < 500:
                raise   # client error — don't retry
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    Server error. Waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)

        except (RateLimitError, APIConnectionError):
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    Rate limited / connection error. Waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)


async def demo_backoff():
    print("\n--- DEMO 2: Exponential Backoff ---")

    # Flaky call — fails 2 times then succeeds
    print("\nFlaky API (fails first 2 attempts, succeeds on 3rd):")
    try:
        result = await call_with_backoff("What is async?", fail_mode="flaky")
        print(f"    Result: {result}")
    except Exception as e:
        print(f"    Failed: {e}")

    # Permanent failure — stops immediately
    print("\nAuth error (permanent — stops immediately, no retry):")
    try:
        await call_with_backoff("What is async?", fail_mode="auth", max_retries=3)
    except AuthenticationError as e:
        print(f"    ✅ Correctly stopped immediately: {e}")


# -------------------------------------------------------
# DEMO 3 — Batch eval with error handling
# Some calls fail, the rest continue
# -------------------------------------------------------

async def demo_batch_error_handling():
    print("\n--- DEMO 3: Batch Eval With Error Handling ---")
    print("Some calls fail — the batch continues, failures are logged")

    test_cases = [
        ("tc_001", "What is async?",      "none"),
        ("tc_002", "What is a token?",    "rate_limit"),   # will fail
        ("tc_003", "What is RAG?",        "none"),
        ("tc_004", "What is top-p?",      "server_error"), # will fail
        ("tc_005", "What is a context?",  "none"),
    ]

    results = []
    errors  = []
    sem = asyncio.Semaphore(2)

    async def safe_call(test_id: str, prompt: str, fail_mode: str):
        async with sem:
            try:
                text = await call_with_backoff(prompt, fail_mode=fail_mode, max_retries=2)
                results.append({"test_id": test_id, "status": "pass", "text": text})
                print(f"  ✅ {test_id} passed")
            except Exception as e:
                errors.append({"test_id": test_id, "error": str(e)})
                print(f"  ❌ {test_id} failed: {type(e).__name__}")

    await asyncio.gather(*[
        safe_call(tid, prompt, fail_mode)
        for tid, prompt, fail_mode in test_cases
    ])

    print(f"\n  Results: {len(results)} passed, {len(errors)} failed")
    if errors:
        print("  Failed test cases:")
        for e in errors:
            print(f"    {e['test_id']}: {e['error']}")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 50)
    print("ERROR HANDLING")
    print("=" * 50)

    await demo_error_types()
    await demo_backoff()
    await demo_batch_error_handling()

    print("\n" + "=" * 50)
    print("Key takeaways:")
    print("  1. RateLimitError   → wait and retry (exponential backoff)")
    print("  2. APIStatusError 5xx → retry with backoff")
    print("  3. APIStatusError 4xx → fix your request, don't retry")
    print("  4. AuthenticationError → check .env, don't retry")
    print("  5. In batch runs, catch per-call — don't let one failure kill the batch")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
