# Real-world async fetcher with rate limiting and error handling.
# Builds on RateLimit.py by adding typed results and per-request error handling
# so one bad URL never crashes the entire run.

import asyncio
import httpx
import time
from dataclasses import dataclass

# ── result types ──────────────────────────────────────────────────
# Instead of returning raw strings or raising exceptions, every fetch
# returns a structured object — either Success or Failure.
# This makes it easy to separate and process results at the end.

@dataclass
class Success:
    url: str
    status: int  # HTTP status code e.g. 200
    bytes: int   # size of the response body

@dataclass
class Failure:
    url: str
    reason: str  # human-readable description of what went wrong

Result = Success | Failure  # every fetch() call returns one of these two — never raises

# ── core fetch ────────────────────────────────────────────────────

async def fetch(
    client: httpx.AsyncClient,  # shared across all requests to reuse the connection pool
    sem: asyncio.Semaphore,     # shared gate that limits how many requests run at once
    url: str,
) -> Result:
    async with sem:  # blocks here if 5 requests are already in-flight; resumes when one finishes
        try:
            r = await client.get(url)
            r.raise_for_status()  # converts 4xx/5xx responses into exceptions so they hit the except blocks below
            return Success(url=url, status=r.status_code, bytes=len(r.content))

        except httpx.TimeoutException:
            # request exceeded the 5s timeout set on the client — treated as a soft failure, not a crash
            return Failure(url=url, reason="timeout")

        except httpx.HTTPStatusError as e:
            # server responded but with an error status (404, 500, 429, etc.)
            return Failure(url=url, reason=f"HTTP {e.response.status_code}")

        except httpx.RequestError as e:
            # network-level failure before a response was received — e.g. DNS lookup failed
            return Failure(url=url, reason=f"request error: {type(e).__name__}")
    # semaphore slot is released here automatically, letting the next waiting coroutine proceed

# ── entry point ───────────────────────────────────────────────────

async def main():
    urls = [
        # mix of fast, slow, and intentionally broken URLs to exercise all code paths
        "https://httpbin.org/get",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",       # triggers HTTPStatusError (not found)
        "https://httpbin.org/status/500",       # triggers HTTPStatusError (server error)
        "https://httpbin.org/delay/1",          # responds after 1s — tests slow requests
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
        "https://httpbin.org/bytes/1024",       # returns 1KB of random bytes
        "https://httpbin.org/bytes/4096",       # returns 4KB of random bytes
        "https://httpbin.org/json",
        "https://httpbin.org/uuid",
        "https://httpbin.org/ip",
        "https://httpbin.org/user-agent",
        "https://httpbin.org/headers",
        "https://httpbin.org/get?page=1",
        "https://httpbin.org/get?page=2",
        "https://httpbin.org/get?page=3",
        "https://httpbin.org/get?page=4",
        "https://not-a-real-domain-xyz.io/",    # triggers RequestError — DNS will fail
        "https://httpbin.org/status/429",       # triggers HTTPStatusError — simulates rate limit
    ]

    sem = asyncio.Semaphore(5)  # at most 5 requests in-flight at once — change this to tune concurrency

    async with httpx.AsyncClient(timeout=5.0) as client:  # any request taking over 5s is abandoned
        t0 = time.perf_counter()  # start the clock to measure total elapsed time

        raw = await asyncio.gather(
            *[fetch(client, sem, url) for url in urls],  # registers all 20 coroutines; semaphore controls actual concurrency
            return_exceptions=True,  # if fetch() itself has an unexpected bug, capture it as a value instead of crashing
        )

        elapsed = time.perf_counter() - t0  # stop the clock after all fetches complete

    # ── process results ───────────────────────────────────────────
    # fetch() handles all expected errors internally, so raw should only contain Success/Failure objects.
    # The isinstance(item, Exception) check is a safety net for any unhandled bugs in fetch() itself.

    results: list[Result] = []
    for item in raw:
        if isinstance(item, Exception):
            # should not happen in normal operation — means fetch() has an unhandled code path
            results.append(Failure(url="unknown", reason=f"unhandled: {item}"))
        else:
            results.append(item)

    successes = [r for r in results if isinstance(r, Success)]  # filter to only successful fetches
    failures  = [r for r in results if isinstance(r, Failure)]  # filter to only failed fetches

    print(f"\n{'─'*52}")
    print(f"  fetched {len(urls)} URLs in {elapsed:.2f}s")
    print(f"  concurrency cap: 5  │  timeout: 5s")
    print(f"{'─'*52}\n")

    print(f"  ✓ {len(successes)} succeeded\n")
    for r in successes:
        print(f"    {r.status}  {r.bytes:>6} bytes  {r.url}")

    print(f"\n  ✗ {len(failures)} failed\n")
    for r in failures:
        print(f"    [{r.reason}]  {r.url}")

    print(f"\n{'─'*52}")

asyncio.run(main())  # starts the async event loop and runs main()