import asyncio
import uuid
import hashlib
from evalforge.models import TestCase, EvalRun, EvalResult
from evalforge.client import client, APIResponse
from evalforge.config import config
from evalforge.scorer import score_output

MODEL_TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are Meridian, a fintech assistant. Be brief and direct. "
    "Avoid unnecessary hedging — get straight to the point. "
    "However, required regulatory disclosures are never optional: "
    "always state investment risk and past-performance disclaimers "
    "when discussing returns, distinguish FDIC vs SIPC coverage "
    "when discussing account insurance, and never give a direct "
    "buy/sell recommendation — redirect personalized advice "
    "questions to a licensed financial advisor.\n\n"
    "Meridian product facts: uninvested cash is held in the Meridian "
    "cash sweep account at partner banks and is FDIC insured up to "
    "$250,000 per depositor, per bank. Investment holdings (stocks, "
    "ETFs, funds) are NOT FDIC insured; they are protected by SIPC up "
    "to $500,000."
)


async def get_model_output(test_case: TestCase) -> APIResponse:
    result = await client.call(
        prompt=test_case.prompt,
        system=SYSTEM_PROMPT,
        temperature=MODEL_TEMPERATURE
    )
    return result

async def run_eval_case(
    test_case: TestCase,
    sem: asyncio.Semaphore
) -> EvalResult:

    async with sem:
        # A failed model call is recorded on its own result so one bad
        # test case (rate limit, provider outage) doesn't abort the batch.
        try:
            response = await get_model_output(test_case)
        except Exception as e:
            return EvalResult(
                test_id=test_case.id,
                model_output="",
                score=0.0,
                reasoning="",
                passed=False,
                error=f"Model call failed: {e}"
            )
        result = await score_output(test_case, response.text)
        result.input_tokens  = response.input_tokens
        result.output_tokens = response.output_tokens
        result.cost_usd      = response.cost_usd
        return result
    

async def run_all(
    test_cases: list,
    run_id: str = None
) -> EvalRun:

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    sem     = asyncio.Semaphore(config.max_concurrent_calls)
    results = await asyncio.gather(*[
        run_eval_case(tc, sem)
        for tc in test_cases
    ])

    model_label = f"mock-{config.model}" if config.mock_mode else config.model

    run = EvalRun(
        run_id=run_id,
        model_version=model_label,
        temperature=MODEL_TEMPERATURE,
        prompt_hash=hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8],
        judge_model=config.judge_model
    )
    run.results = list(results)
    run.compute_stats()
    return run