CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cider_score NUMERIC(5, 4) NOT NULL,
    hallucination_rate NUMERIC(5, 4) NOT NULL,
    deployment_status VARCHAR(50) NOT NULL
);