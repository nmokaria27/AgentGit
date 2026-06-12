"""Run once to create the telemetry table in ClickHouse Cloud."""
import os
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

DDL = """
CREATE TABLE IF NOT EXISTS gitagent_telemetry (
    timestamp    DateTime64(3) DEFAULT now64(),
    session_id   String,
    step_count   Int32,
    current_thought  String,
    previous_thought String,
    exit_code    Int32,
    decision     String,
    rollback_count Int32
) ENGINE = MergeTree()
ORDER BY (timestamp, step_count)
"""

if __name__ == "__main__":
    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ["CLICKHOUSE_PASSWORD"],
        secure=True,
    )
    client.command(DDL)
    print("Table gitagent_telemetry created (or already exists).")
