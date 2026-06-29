import asyncio
from evalforge.models import load_test_cases
from evalforge.runner import run_all
from evalforge.storage import get_pool, save_run


async def main():
    pool       = await get_pool()
    test_cases = load_test_cases("test_cases/phase3.yaml")

    print(f"Running {len(test_cases)} test cases...")
    run = await run_all(test_cases)
    await save_run(run, pool)
    await pool.close()

    print(f"\nRun ID    : {run.run_id}")
    print(f"Pass rate : {run.pass_rate * 100:.0f}%")
    print(f"Avg score : {run.avg_score:.1f}/5.0")
    print(f"Cost      : ${run.total_cost:.6f}")
    print("\nResults:")
    for r in run.results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} {r.test_id} | score: {r.score} | {r.reasoning[:50]}")


if __name__ == "__main__":
    asyncio.run(main())