import asyncio
import sys
from evalforge.models import load_test_cases
from evalforge.runner import run_all
from evalforge.storage import get_pool, save_run, load_baseline, export_dashboard_data
from evalforge.config import config, ConfigError


async def run_regression(baseline_run_id: str = None) -> bool:
    test_cases = load_test_cases("test_cases/meridian_advisor.yaml")
    test_ids   = [tc.id for tc in test_cases]

    mode = "DEMO (mock, offline)" if config.mock_mode else f"LIVE ({config.provider}: {config.model})"
    print(f"Mode: {mode}")
    print(f"Running {len(test_cases)} test cases...")
    run = await run_all(test_cases)

    print(f"\nRun complete: {run.run_id}")
    print(f"Pass rate  : {run.pass_rate * 100:.0f}%")
    print(f"Avg score  : {run.avg_score:.1f}/5.0")
    print(f"Total cost : ${run.total_cost:.6f}")
    print("\nResults:")
    for r in run.results:
        if r.error:
            print(f"  ⚠️  {r.test_id} | error: {r.error[:80]}")
        else:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} {r.test_id} | score: {r.score} | {r.reasoning[:60]}")

    # An errored test case has no score, so the regression gate can't vouch
    # for it. Fail loudly instead of reporting "no regressions".
    errored = [r for r in run.results if r.error]
    if errored:
        print(f"\n❌ {len(errored)} test case(s) errored — run not saved, failing.")
        return False

    if not config.use_database:
        print("\nDemo mode: skipping database save and baseline comparison.")
        print("✅ Pipeline completed successfully.")
        return True

    pool = await get_pool()
    await save_run(run, pool)

    if baseline_run_id is None:
        print("\nNo baseline set — this run will be the baseline.")
        print(f"Set BASELINE_RUN_ID={run.run_id} in .env to use it.")
        await export_dashboard_data(pool, run.run_id, test_ids=test_ids)
        await pool.close()
        return True

    print(f"\nComparing to baseline: {baseline_run_id}")
    baseline = await load_baseline(baseline_run_id, pool)

    if not baseline:
        print("Baseline run not found in database.")
        await export_dashboard_data(pool, run.run_id, test_ids=test_ids)
        await pool.close()
        return True

    regressions = []
    for result in run.results:
        if result.test_id not in baseline:
            continue
        drop = baseline[result.test_id] - result.score
        if drop >= 1.0:
            regressions.append({
                "test_id":        result.test_id,
                "baseline_score": baseline[result.test_id],
                "current_score":  result.score,
                "drop":           round(drop, 1),
                "reasoning":      result.reasoning
            })

    if regressions:
        print(f"\n❌ {len(regressions)} regression(s) detected:")
        for r in regressions:
            print(f"  {r['test_id']}: {r['baseline_score']} → {r['current_score']} (drop {r['drop']})")
            print(f"  Reason: {r['reasoning']}")
    else:
        print("\n✅ No regressions detected.")

    await export_dashboard_data(pool, baseline_run_id, test_ids=test_ids)
    await pool.close()
    return len(regressions) == 0


async def main():
    try:
        config.validate_live()
    except ConfigError as e:
        print(f"Config error: {e}")
        sys.exit(2)
    passed = await run_regression(config.baseline_run_id)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
