import psycopg2
from datetime import datetime

DB_PARAMS = {
    "dbname": "vlm_metrics",
    "user": "root",
    "password": "1234",
    "host": "localhost",
    "port": "5433"
}

def log_evaluation(model_version, similarity_score, hallucination_rate, deployment_status, mmd_p_value=None):
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        query = """
        INSERT INTO evaluations (model_version, mmd_p_value, similarity_score, hallucination_rate, deployment_status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """
        values = (model_version, mmd_p_value, similarity_score, hallucination_rate, deployment_status, datetime.now())
        
        cur.execute(query, values)
        row_id = cur.fetchone()[0]
        conn.commit()
        
        print(f"✅ SUCCESS: Logged {model_version} to database!")
        print(f"   -> DB Row ID: {row_id}")
        print(f"   -> Timestamp: {values[5]}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")