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
            SELECT run_id, model_version, pass_rate, avg_score,
                   total_cost, created_at
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