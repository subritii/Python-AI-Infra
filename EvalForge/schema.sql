-- Postgres schema for live runs (demo mode does not use a database).
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id        TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    temperature   DOUBLE PRECISION NOT NULL,
    prompt_hash   TEXT,
    pass_rate     DOUBLE PRECISION DEFAULT 0.0,
    avg_score     DOUBLE PRECISION DEFAULT 0.0,
    total_cost    DOUBLE PRECISION DEFAULT 0.0,
    created_at    TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_results (
    id            SERIAL PRIMARY KEY,
    run_id        TEXT REFERENCES eval_runs(run_id),
    test_id       TEXT NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    passed        BOOLEAN NOT NULL,
    reasoning     TEXT,
    issues        TEXT[],
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      DOUBLE PRECISION DEFAULT 0.0,
    created_at    TIMESTAMP DEFAULT now()
);
