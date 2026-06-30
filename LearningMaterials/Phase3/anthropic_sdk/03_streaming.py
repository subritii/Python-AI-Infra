# phase3/01_anthropic_sdk/03_streaming.py
# -------------------------------------------------------
# TOPIC: Streaming responses
#
# What you'll learn:
#   - Sync streaming with client.messages.stream()
#   - Async streaming with AsyncAnthropic
#   - How to collect streamed tokens into a full string
#   - When to stream vs when not to
#
# Run: python 03_streaming.py
# -------------------------------------------------------

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


# -------------------------------------------------------
# Mock streaming — yields tokens one by one with delay
# Simulates real streaming behavior
# -------------------------------------------------------

async def mock_stream_tokens(text: str, delay: float = 0.05):
    """Yields tokens one at a time with a small delay — like real streaming."""
    words = text.split()
    for word in words:
        yield word + " "
        await asyncio.sleep(delay)


# -------------------------------------------------------
# DEMO 1 — Non-streaming vs streaming side by side
# -------------------------------------------------------

async def demo_streaming_vs_not():
    print("\n--- DEMO 1: Non-streaming vs Streaming ---")

    fake_response = (
        "The async keyword in Python marks a function as a coroutine. "
        "When you call an async function, it returns a coroutine object. "
        "You need to await it to actually run it and get the result."
    )

    # Non-streaming — you wait, then get everything at once
    print("\nNon-streaming (wait for full response):")
    start = time.perf_counter()
    await asyncio.sleep(0.5)    # simulates waiting for full response
    elapsed = time.perf_counter() - start
    print(f"  [{elapsed:.2f}s wait]  {fake_response}")

    # Streaming — tokens appear as they're generated
    print("\nStreaming (tokens appear in real time):")
    print("  ", end="", flush=True)
    start = time.perf_counter()
    full_text = ""
    async for token in mock_stream_tokens(fake_response, delay=0.04):
        print(token, end="", flush=True)
        full_text += token
    elapsed = time.perf_counter() - start
    print(f"\n  [{elapsed:.2f}s total, but user sees output immediately]")


# -------------------------------------------------------
# DEMO 2 — Async streaming (EvalForge pattern)
# Collects streamed tokens into a full string
# -------------------------------------------------------

async def stream_and_collect(prompt: str) -> tuple[str, int]:
    """
    Streams a response and collects all tokens.
    Returns (full_text, approximate_output_tokens).
    
    In real code, replace mock_stream_tokens with:
    
    async with async_client.messages.stream(...) as stream:
        async for text in stream.text_stream:
            full_text += text
    """
    print(f"\n  Streaming response for: '{prompt[:50]}'")
    print("  ", end="", flush=True)

    full_text = ""
    fake_response = f"This is a streaming response about: {prompt}. It arrives token by token."

    async for token in mock_stream_tokens(fake_response, delay=0.03):
        print(token, end="", flush=True)
        full_text += token

    token_count = len(full_text.split())
    print(f"\n  Complete. {token_count} tokens received.")
    return full_text, token_count


async def demo_async_streaming():
    print("\n--- DEMO 2: Async Streaming Pattern ---")

    result, tokens = await stream_and_collect("What is temperature in LLMs?")
    print(f"\n  Full text length: {len(result)} chars")
    print(f"  Token count: {tokens}")


# -------------------------------------------------------
# DEMO 3 — Real streaming code (shown but not run)
# This is what you'll write when MOCK_MODE=false
# -------------------------------------------------------

def show_real_streaming_code():
    print("\n--- DEMO 3: Real Streaming Code (reference) ---")
    print("""
  # Sync streaming
  with client.messages.stream(
      model="claude-sonnet-4-6",
      max_tokens=1024,
      messages=[{"role": "user", "content": prompt}]
  ) as stream:
      for text in stream.text_stream:
          print(text, end="", flush=True)
  
  # Get final message object (has usage stats)
  final = stream.get_final_message()
  print(f"Tokens: {final.usage.input_tokens} in, {final.usage.output_tokens} out")


  # Async streaming (use this in EvalForge)
  async def stream_response(prompt: str) -> str:
      full_text = ""
      async with async_client.messages.stream(
          model="claude-sonnet-4-6",
          max_tokens=1024,
          messages=[{"role": "user", "content": prompt}]
      ) as stream:
          async for text in stream.text_stream:
              full_text += text
              print(text, end="", flush=True)
      
      final = await stream.get_final_message()
      return full_text, final.usage
""")


# -------------------------------------------------------
# DEMO 4 — When to stream vs not
# -------------------------------------------------------

def demo_when_to_stream():
    print("\n--- DEMO 4: When to stream vs not ---")
    print("""
  STREAM when:
    ✅ Dashboard showing live eval progress
    ✅ Long judge responses (don't wait 10s for a rubric)
    ✅ Any user-facing output — instant feedback feels better
    ✅ Debugging — see where generation goes wrong in real time

  DON'T stream when:
    ❌ Batch eval runs where you process all results after
    ❌ Short responses (< 100 tokens) — overhead not worth it
    ❌ Parallel calls with gather — streaming + gather is complex
    ❌ When you need the full text before doing anything with it

  EvalForge rule:
    - eval runner    → no streaming (batch, process after)
    - dashboard feed → stream (live progress display)
""")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 50)
    print("STREAMING")
    print("=" * 50)
    print(f"MOCK_MODE: {MOCK_MODE}")

    await demo_streaming_vs_not()
    await demo_async_streaming()
    show_real_streaming_code()
    demo_when_to_stream()

    print("=" * 50)
    print("Key takeaways:")
    print("  1. stream.text_stream yields tokens one at a time")
    print("  2. get_final_message() gives usage stats after streaming")
    print("  3. Use async streaming in EvalForge dashboard")
    print("  4. Don't stream in batch eval runner — adds complexity")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
