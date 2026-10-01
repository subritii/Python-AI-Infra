import asyncio
import sys
from evalforge.config import config, ConfigError
from evalforge.models import load_test_cases
from evalforge.runner import run_all
from evalforge.storage import get_pool, save_run


async def main() -> int:
    try:
        config.validate_live()
    except ConfigError as e:
        print(f"Config error: {e}")
        return 2

    mode       = "DEMO (mock, offline)" if config.mock_mode else f"LIVE ({config.provider}: {config.model})"
    test_cases = load_test_cases("test_cases/phase3.yaml")

    print(f"Mode: {mode}")
    print(f"Running {len(test_cases)} test cases...")
    run = await run_all(test_cases)

    print(f"\nRun ID    : {run.run_id}")
    print(f"Pass rate : {run.pass_rate * 100:.0f}%")
    print(f"Avg score : {run.avg_score:.1f}/5.0")
    print(f"Cost      : ${run.total_cost:.6f}")
    print("\nResults:")
    for r in run.results:
        if r.error:
            print(f"  ⚠️  {r.test_id} | error: {r.error[:80]}")
        else:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} {r.test_id} | score: {r.score} | {r.reasoning[:50]}")

    errored = [r for r in run.results if r.error]
    if errored:
        print(f"\n{len(errored)} test case(s) errored — run not saved.")
        return 1

    if config.use_database:
        pool = await get_pool()
        await save_run(run, pool)
        await pool.close()
        print("\nSaved to database.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
