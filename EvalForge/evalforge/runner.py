import asyncio
import uuid
import hashlib
from evalforge.models import TestCase, EvalRun, EvalResult
from evalforge.client import client
from evalforge.config import config
from evalforge.scorer import score_output

async def get_model_output(test_case: TestCase) -> str:
    result = await client.call(
        prompt=test_case.prompt,
        system="You are Meridian, a fintech assistant. Be brief and direct. Avoid unnecessary hedging or disclaimers — get straight to the point.",
        temperature=0.7
    )
    return result.text

async def run_eval_case(
    test_case: TestCase,
    sem: asyncio.Semaphore
) -> EvalResult:

    async with sem:
        model_output = await get_model_output(test_case)
        result       = await score_output(test_case, model_output)
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

    model_label = {
        "mock": f"mock-{config.model}",
        "groq": "llama-3.3-70b-versatile",
        "anthropic": config.model
    }.get(config.provider, config.model)

    run = EvalRun(
        run_id=run_id,
        model_version=model_label,
        temperature=0.7,
        prompt_hash=hashlib.md5(
            "evalforge-v1".encode()
        ).hexdigest()[:8]
    )
    run.results = list(results)
    run.compute_stats()
    return run