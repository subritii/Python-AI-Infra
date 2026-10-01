
import json
import os
import asyncpg
from evalforge.models import EvalRun, EvalResult
from evalforge.config import config

async def get_pool():
    return await asyncpg.create_pool(
        config.database_url,
        min_size=2,
        max_size=10
    )


async def save_run(run: EvalRun, pool) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                INSERT INTO eval_runs
                    (run_id, model_version, temperature, prompt_hash,
                     pass_rate, avg_score, total_cost)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            run.run_id, run.model_version, run.temperature,
            run.prompt_hash, run.pass_rate, run.avg_score, run.total_cost
            )

            for result in run.results:
                if result.error is None:
                    await conn.execute("""
                        INSERT INTO eval_results
                            (run_id, test_id, score, passed, reasoning,
                             issues, input_tokens, output_tokens, cost_usd)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    run.run_id, result.test_id, result.score,
                    result.passed, result.reasoning, result.issues,
                    result.input_tokens, result.output_tokens, result.cost_usd
                    )

async def load_baseline(run_id: str, pool) -> dict:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT test_id, score
            FROM eval_results
            WHERE run_id = $1
        """, run_id)
        return {row["test_id"]: row["score"] for row in rows}


async def get_recent_runs(limit: int, pool) -> list:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT run_id, model_version, temperature, prompt_hash,
                   pass_rate, avg_score, total_cost, created_at
            FROM eval_runs
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)
        return [dict(row) for row in rows]


async def get_run_results(run_id: str, pool) -> list:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT test_id, score, passed, reasoning
            FROM eval_results
            WHERE run_id = $1
            ORDER BY test_id
        """, run_id)
        return [dict(row) for row in rows]


async def get_results_for_runs(run_ids: list, pool) -> dict:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT run_id, test_id, score, passed, reasoning
            FROM eval_results
            WHERE run_id = ANY($1::text[])
            ORDER BY test_id
        """, run_ids)
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append({
            "test_id":   row["test_id"],
            "score":     row["score"],
            "passed":    row["passed"],
            "reasoning": row["reasoning"],
        })
    return by_run


async def export_dashboard_data(
    pool,
    baseline_run_id: str,
    output_path: str = "docs/dashboard_data.json",
    test_ids: list = None,
    limit: int = 50
):
    recent_runs = await get_recent_runs(limit, pool)
    baseline    = await load_baseline(baseline_run_id, pool)
    results     = await get_results_for_runs([r["run_id"] for r in recent_runs], pool)

    # eval_runs is shared by every suite; keep only runs of the suite being exported.
    if test_ids is not None:
        wanted      = set(test_ids)
        recent_runs = [
            r for r in recent_runs
            if any(x["test_id"] in wanted for x in results.get(r["run_id"], []))
        ]

    if not recent_runs:
        return

    latest = recent_runs[0]
    latest_results = results.get(latest["run_id"], [])

    regressions = []
    for r in latest_results:
        if r["test_id"] in baseline:
            drop = baseline[r["test_id"]] - r["score"]
            if drop >= 1.0:
                regressions.append({
                    "test_id":        r["test_id"],
                    "baseline_score": baseline[r["test_id"]],
                    "current_score":  r["score"],
                    "drop":           round(drop, 1),
                    "reasoning":      r["reasoning"]
                })

    data = {
        "generated_at": str(latest["created_at"]),
        "baseline_run_id": baseline_run_id,
        "status": {
            "latest_run_id": latest["run_id"],
            "model_version": latest["model_version"],
            "pass_rate": latest["pass_rate"],
            "avg_score": latest["avg_score"],
            "total_cost": latest["total_cost"],
        },
        "runs": [
            {
                "run_id": r["run_id"],
                "created_at": str(r["created_at"]),
                "model_version": r["model_version"],
                "temperature": r["temperature"],
                "prompt_hash": r["prompt_hash"],
                "pass_rate": r["pass_rate"],
                "avg_score": r["avg_score"],
                "total_cost": r["total_cost"],
                "results": results.get(r["run_id"], []),
            }
            for r in recent_runs
        ],
        "active_regressions": regressions
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)