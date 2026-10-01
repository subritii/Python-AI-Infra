import json
import re
import asyncio
from dataclasses import dataclass
from evalforge.config import config

@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    stop_reason: str

class EvalForgeClient:

    INPUT_COST  = 0.000003
    OUTPUT_COST = 0.000015

    # Rate-limit (429) and transient errors are retried by the SDKs with exponential
    # backoff that honours Retry-After; Groq's free tier allows 8k tokens/min per model.
    MAX_RETRIES = 8
    # The SDK's backoff tops out in seconds; per-minute token windows can need longer.
    RATE_LIMIT_RETRIES = 4

    def __init__(self, provider: str, model: str):
        self.mock_mode = config.mock_mode
        self.model     = model
        self.provider  = provider
        self._client   = None

        # Missing keys are reported by config.validate_live() before any call.
        if not self.mock_mode and self.provider == "anthropic" and config.anthropic_api_key:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=config.anthropic_api_key, max_retries=self.MAX_RETRIES
            )
        elif not self.mock_mode and self.provider == "groq" and config.groq_api_key:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=config.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                max_retries=self.MAX_RETRIES
            )

    def _mock_call(self, prompt: str, system: str = "") -> APIResponse:
        if "score this output" in prompt.lower():
            text = json.dumps({
                "score": 4.2,
                "reasoning": "Mock: correct and clear explanation.",
                "passed": True,
                "issues": []
            })
        else:
            text = f"Mock response for: {prompt[:60]}"

        input_tokens  = (len(prompt) + len(system)) // 4
        output_tokens = len(text) // 4

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(input_tokens * self.INPUT_COST) + (output_tokens * self.OUTPUT_COST),
            model=f"mock-{self.model}",
            stop_reason="end_turn"
        )
    
    async def _real_call(
        self, prompt: str, system: str = "",
        temperature: float = 0.0, max_tokens: int = 1024
    ) -> APIResponse:

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        if system:
            kwargs["system"] = system

        # **kwargs unpacks the dictionary into keyword arguments. 
        response = await self._client.messages.create(**kwargs)

        it = response.usage.input_tokens
        ot = response.usage.output_tokens

        return APIResponse(
            text=response.content[0].text,
            input_tokens=it,
            output_tokens=ot,
            cost_usd=(it * self.INPUT_COST) + (ot * self.OUTPUT_COST),
            model=response.model,
            stop_reason=response.stop_reason
        )
    
    async def _groq_call(
        self, prompt: str, system: str = "",
        temperature: float = 0.0, max_tokens: int = 1024
    ) -> APIResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        choice = response.choices[0]
        text   = choice.message.content
        if not text:
            raise RuntimeError(f"Groq returned an empty response (finish_reason={choice.finish_reason})")
        input_tokens  = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            model=response.model,
            stop_reason=choice.finish_reason
        )
    
    async def call(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> APIResponse:

        if self.mock_mode:
            return self._mock_call(prompt, system)

        for attempt in range(self.RATE_LIMIT_RETRIES + 1):
            try:
                if self.provider == "groq":
                    return await self._groq_call(prompt, system, temperature, max_tokens)
                return await self._real_call(prompt, system, temperature, max_tokens)
            except Exception as e:
                if getattr(e, "status_code", None) != 429 or attempt == self.RATE_LIMIT_RETRIES:
                    raise
                await asyncio.sleep(self._retry_after(e))

    @staticmethod
    def _retry_after(error: Exception) -> float:
        # Groq says e.g. "Please try again in 30.754s"; fall back to a full minute window.
        match = re.search(r"try again in ([\d.]+)s", str(error))
        return float(match.group(1)) + 1 if match else 60.0
    

client       = EvalForgeClient(config.provider, config.model)              # model under test
judge_client = EvalForgeClient(config.judge_provider, config.judge_model)  # LLM judge