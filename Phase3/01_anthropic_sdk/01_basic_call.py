# -------------------------------------------------------
# TOPIC: Basic Anthropic API call
#
# What you'll learn:
#   - How to create a client
#   - How to send a message
#   - What the response object looks like
#   - How to read tokens and cost
#
# Run: python 01_basic_call.py
# Requires: MOCK_MODE=true in .env (no API key needed)
# -------------------------------------------------------

import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


# -------------------------------------------------------
# The response shape — same whether real or mocked
# -------------------------------------------------------

@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    stop_reason: str

    def summary(self) -> str:
        return (
            f"model       : {self.model}\n"
            f"stop_reason : {self.stop_reason}\n"
            f"input_tokens: {self.input_tokens}\n"
            f"output_tokens:{self.output_tokens}\n"
            f"cost_usd    : ${self.cost_usd:.6f}\n"
            f"text        : {self.text[:100]}..."
        )


# -------------------------------------------------------
# Mock response — simulates what Anthropic returns
# -------------------------------------------------------

def mock_response(prompt: str) -> APIResponse:
    """
    Returns a fake but realistic response.
    Shape is identical to a real API response.
    """
    fake_text = json.dumps({
        "score": 4.2,
        "reasoning": "The output correctly identifies async def as marking a coroutine.",
        "passed": True,
        "issues": []
    })

    # Rough token estimate: chars / 4
    input_tokens  = len(prompt) // 4
    output_tokens = len(fake_text) // 4

    return APIResponse(
        text=fake_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=(input_tokens * 0.000003) + (output_tokens * 0.000015),
        model="mock-claude-sonnet",
        stop_reason="end_turn"
    )


# -------------------------------------------------------
# Real response — only runs if MOCK_MODE=false
# -------------------------------------------------------

def real_response(prompt: str, system: str = "") -> APIResponse:
    import anthropic
    client = anthropic.Anthropic()

    kwargs = dict(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return APIResponse(
        text=response.content[0].text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=(input_tokens * 0.000003) + (output_tokens * 0.000015),
        model=response.model,
        stop_reason=response.stop_reason
    )


# -------------------------------------------------------
# Unified call — picks mock or real based on MOCK_MODE
# -------------------------------------------------------

def call_api(prompt: str, system: str = "") -> APIResponse:
    if MOCK_MODE:
        print("[MOCK MODE] Returning fake response — set MOCK_MODE=false for real API")
        return mock_response(prompt)
    return real_response(prompt, system)


# -------------------------------------------------------
# Run it
# -------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("BASIC API CALL")
    print("=" * 50)

    # 1. Simple call
    print("\n--- 1. Simple call ---")
    prompt = "What does async def do in Python? Answer in one sentence."
    result = call_api(prompt)
    print(result.summary())

    # 2. Call with system prompt
    print("\n--- 2. With system prompt ---")
    result2 = call_api(
        prompt="Score this: 'async def marks a coroutine'",
        system="You are an eval judge. Return JSON with score (1-5) and reasoning."
    )
    print(result2.summary())

    # 3. Inspect the response fields
    print("\n--- 3. Response fields ---")
    print(f"text         : {result.text}")
    print(f"input_tokens : {result.input_tokens}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"cost_usd     : ${result.cost_usd:.6f}")
    print(f"model        : {result.model}")
    print(f"stop_reason  : {result.stop_reason}")
