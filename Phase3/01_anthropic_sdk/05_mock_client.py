# phase3/01_anthropic_sdk/05_mock_client.py
# -------------------------------------------------------
# TOPIC: Complete EvalForge client with mock mode
#
# What you'll learn:
#   - Production-ready client that switches between mock/real
#   - Multi-turn conversation handling
#   - Token counting before sending
#   - The exact client you'll use in Phase 4
#
# Run: python 05_mock_client.py
# -------------------------------------------------------

import asyncio
import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


# -------------------------------------------------------
# Data shapes
# -------------------------------------------------------

@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    stop_reason: str


@dataclass
class Message:
    role: str       # "user" or "assistant"
    content: str


# -------------------------------------------------------
# The EvalForge client
# Drop this into evalforge/client.py in Phase 4
# -------------------------------------------------------

class EvalForgeClient:
    """
    Unified Anthropic client for EvalForge.
    
    MOCK_MODE=true  → returns fake responses (free, fast, offline)
    MOCK_MODE=false → calls real Anthropic API
    
    Usage:
        client = EvalForgeClient()
        result = await client.call("Your prompt here")
        print(result.text, result.cost_usd)
    """

    INPUT_COST_PER_TOKEN  = 0.000003    # $3 per million
    OUTPUT_COST_PER_TOKEN = 0.000015    # $15 per million
    DEFAULT_MODEL         = "claude-sonnet-4-6"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        mock_mode: bool = MOCK_MODE
    ):
        self.model = model
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self._client = None

        if not mock_mode:
            import anthropic
            self._client = anthropic.AsyncAnthropic()

    # -------------------------------------------------------
    # Core call method
    # -------------------------------------------------------

    async def call(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        messages: Optional[list[Message]] = None
    ) -> APIResponse:
        """
        Make a single API call.
        
        Args:
            prompt:      The user message
            system:      System prompt (optional)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens:  Max response tokens
            messages:    Full conversation history (optional — for multi-turn)
        """
        if self.mock_mode:
            return self._mock_call(prompt, system)
        return await self._real_call(prompt, system, temperature, max_tokens, messages)

    # -------------------------------------------------------
    # Mock implementation
    # -------------------------------------------------------

    def _mock_call(self, prompt: str, system: str = "") -> APIResponse:
        """Returns a realistic fake response based on prompt content."""
        # Generate different mock responses based on prompt type
        if "score" in prompt.lower() or "eval" in prompt.lower():
            text = json.dumps({
                "score": 4.2,
                "reasoning": "The output correctly identifies the concept with clear explanation.",
                "passed": True,
                "issues": []
            })
        elif "summary" in prompt.lower():
            text = "Mock summary: The topic covers key concepts including definitions, examples, and practical applications."
        else:
            text = f"Mock response for prompt: '{prompt[:60]}...'"

        input_tokens  = (len(prompt) + len(system)) // 4
        output_tokens = len(text) // 4

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(input_tokens * self.INPUT_COST_PER_TOKEN) +
                     (output_tokens * self.OUTPUT_COST_PER_TOKEN),
            model=f"mock-{self.model}",
            stop_reason="end_turn"
        )

    # -------------------------------------------------------
    # Real implementation
    # -------------------------------------------------------

    async def _real_call(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[list[Message]]
    ) -> APIResponse:
        import anthropic

        # Build messages list
        if messages:
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        else:
            msg_list = [{"role": "user", "content": prompt}]

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msg_list
        )
        if system:
            kwargs["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = await self._client.messages.create(**kwargs)
                it = response.usage.input_tokens
                ot = response.usage.output_tokens
                return APIResponse(
                    text=response.content[0].text,
                    input_tokens=it,
                    output_tokens=ot,
                    cost_usd=(it * self.INPUT_COST_PER_TOKEN) + (ot * self.OUTPUT_COST_PER_TOKEN),
                    model=response.model,
                    stop_reason=response.stop_reason
                )
            except anthropic.RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    # -------------------------------------------------------
    # Token counting (pre-flight check)
    # -------------------------------------------------------

    async def count_tokens(self, prompt: str, system: str = "") -> int:
        """Estimate tokens before sending — avoids surprise costs."""
        if self.mock_mode:
            return (len(prompt) + len(system)) // 4
        result = self._client.messages.count_tokens(
            model=self.model,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}]
        )
        return result.input_tokens


# -------------------------------------------------------
# DEMO 1 — Basic usage
# -------------------------------------------------------

async def demo_basic_usage():
    print("\n--- DEMO 1: Basic Usage ---")
    client = EvalForgeClient()
    print(f"Mock mode: {client.mock_mode}")

    result = await client.call(
        prompt="What does async def do in Python?",
        system="Answer in one sentence."
    )
    print(f"Response : {result.text[:80]}")
    print(f"Tokens   : {result.input_tokens} in / {result.output_tokens} out")
    print(f"Cost     : ${result.cost_usd:.6f}")
    print(f"Model    : {result.model}")


# -------------------------------------------------------
# DEMO 2 — Multi-turn conversation
# -------------------------------------------------------

async def demo_multi_turn():
    print("\n--- DEMO 2: Multi-turn Conversation ---")
    client = EvalForgeClient()

    history: list[Message] = []

    async def chat(user_input: str) -> str:
        history.append(Message(role="user", content=user_input))
        result = await client.call(
            prompt=user_input,
            messages=history,
            system="You are a Python tutor."
        )
        history.append(Message(role="assistant", content=result.text))
        return result.text

    print("Turn 1:")
    r1 = await chat("What is async/await?")
    print(f"  User: What is async/await?")
    print(f"  Bot : {r1[:80]}")

    print("Turn 2:")
    r2 = await chat("Give me an example.")
    print(f"  User: Give me an example.")
    print(f"  Bot : {r2[:80]}")

    print(f"\nHistory has {len(history)} messages — sent on every call")


# -------------------------------------------------------
# DEMO 3 — Batch eval calls (the EvalForge runner pattern)
# -------------------------------------------------------

async def demo_batch_evals():
    print("\n--- DEMO 3: Batch Eval Calls ---")
    client = EvalForgeClient()

    test_cases = [
        ("tc_001", "Score this eval output about async/await"),
        ("tc_002", "Score this eval output about tokenization"),
        ("tc_003", "Score this eval output about context windows"),
        ("tc_004", "Score this eval output about temperature"),
        ("tc_005", "Score this eval output about hallucinations"),
    ]

    sem = asyncio.Semaphore(3)
    results = []

    async def run_one(test_id: str, prompt: str):
        async with sem:
            result = await client.call(prompt=prompt, temperature=0.0)
            results.append({
                "test_id": test_id,
                "cost": result.cost_usd,
                "tokens": result.input_tokens + result.output_tokens
            })

    await asyncio.gather(*[run_one(tid, p) for tid, p in test_cases])

    total_cost   = sum(r["cost"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    print(f"  Ran {len(results)} evals")
    print(f"  Total tokens : {total_tokens}")
    print(f"  Total cost   : ${total_cost:.6f}")
    print(f"  Avg cost/eval: ${total_cost / len(results):.6f}")


# -------------------------------------------------------
# Run all demos
# -------------------------------------------------------

async def main():
    print("=" * 50)
    print("EVALFORGE CLIENT (COMPLETE)")
    print("=" * 50)

    await demo_basic_usage()
    await demo_multi_turn()
    await demo_batch_evals()

    print("\n" + "=" * 50)
    print("This is your Phase 4 client.")
    print("Copy EvalForgeClient into evalforge/client.py")
    print("Set MOCK_MODE=false in .env when ready for real API")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
