DROP TABLE IF EXISTS evaluations;

CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(255) NOT NULL,
    mmd_p_value NUMERIC(6,4),
    similarity_score NUMERIC(6,4),
    hallucination_rate NUMERIC(6,4),
    deployment_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);