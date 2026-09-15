import psycopg2
from psycopg2 import Error

# These credentials match your Docker setup (Port 5433, password 1234)
DB_CONFIG = {
    "host": "localhost",
    "port": "5433", 
    "database": "vlm_metrics",
    "user": "root",
    "password": "1234"
}

def log_evaluation(model_version, cider_score, hallucination_rate, deployment_status):
    """Connects to PostgreSQL and inserts a new evaluation row."""
    connection = None
    try:
        # 1. Establish the connection to PostgreSQL
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # 2. SQL query to insert evaluation metrics into the 'evaluations' table
        insert_query = """
            INSERT INTO evaluations (model_version, cider_score, hallucination_rate, deployment_status)
            VALUES (%s, %s, %s, %s)
            RETURNING id, timestamp;
        """
        
        # 3. Execute the SQL command with values
        cursor.execute(insert_query, (model_version, cider_score, hallucination_rate, deployment_status))
        connection.commit()

        # 4. Retrieve the newly created ID and timestamp
        inserted_row = cursor.fetchone()
        print(f"✅ SUCCESS: Logged {model_version} to database!")
        print(f"   -> DB Row ID: {inserted_row[0]}")
        print(f"   -> Timestamp: {inserted_row[1]}")

    except Error as e:
        print(f"❌ DATABASE ERROR: {e}")
    finally:
        # 5. Safely close database connections
        if connection:
            cursor.close()
            connection.close()

# --- Test Script Execution ---
if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    # Logging test metrics into the database
    log_evaluation(
        model_version="instruct-blip-v1",
        cider_score=0.8542,
        hallucination_rate=0.0210,
        deployment_status="APPROVED"
    )