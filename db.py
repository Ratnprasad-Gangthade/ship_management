import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_values


load_dotenv(override=True)


def _to_db_value(value):
    if pd.isna(value):
        return None
    return value


def _to_db_records(dataframe: pd.DataFrame, columns: list[str]) -> list[tuple]:
    records = []
    for row in dataframe[columns].itertuples(index=False, name=None):
        records.append(tuple(_to_db_value(value) for value in row))
    return records


def _get_db_connection_kwargs() -> Dict[str, Any]:
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if all([host, port, db_name, user, password]):
        return {
            "host": host,
            "port": int(port),
            "dbname": db_name,
            "user": user,
            "password": password,
        }

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return {"dsn": database_url}

    raise ValueError(
        "Database configuration missing. Set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD "
        "(preferred) or DATABASE_URL."
    )


@contextmanager
def get_connection():
    connection_kwargs = _get_db_connection_kwargs()
    connection = psycopg2.connect(**connection_kwargs)
    try:
        yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS oils (
                    oil_id TEXT PRIMARY KEY,
                    oil_type TEXT NOT NULL,
                    delivery_deadline DATE NOT NULL,
                    origin_port TEXT,
                    origin_country TEXT NOT NULL,
                    destination_port TEXT NOT NULL,
                    quantity_mt NUMERIC NOT NULL CHECK (quantity_mt > 0)
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ships (
                    ship_id TEXT PRIMARY KEY,
                    capacity_mt NUMERIC NOT NULL CHECK (capacity_mt > 0),
                    last_oil_type TEXT,
                    available_date DATE NOT NULL,
                    available_port TEXT NOT NULL
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS allocations (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL DEFAULT 0,
                    oil_id TEXT NOT NULL REFERENCES oils(oil_id) ON DELETE CASCADE,
                    ship_id TEXT REFERENCES ships(ship_id) ON DELETE SET NULL,
                    status TEXT NOT NULL CHECK (status IN ('allocated', 'unallocated')),
                    reason TEXT,
                    final_score NUMERIC,
                    allocation_time TIMESTAMP NOT NULL DEFAULT NOW(),
                    allocated_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS allocation_history (
                    id BIGSERIAL PRIMARY KEY,
                    run_id BIGINT NOT NULL,
                    oil_id TEXT NOT NULL,
                    ship_id TEXT,
                    status TEXT NOT NULL CHECK (status IN ('allocated', 'unallocated')),
                    reason TEXT,
                    final_score NUMERIC,
                    allocation_time TIMESTAMP NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                ALTER TABLE oils
                ADD COLUMN IF NOT EXISTS origin_port TEXT;
                """
            )

            cursor.execute(
                """
                UPDATE oils
                SET origin_port = origin_country
                WHERE origin_port IS NULL AND origin_country IS NOT NULL;
                """
            )

            cursor.execute(
                """
                ALTER TABLE allocations
                ADD COLUMN IF NOT EXISTS run_id BIGINT NOT NULL DEFAULT 0;
                """
            )

            cursor.execute(
                """
                ALTER TABLE allocations
                ADD COLUMN IF NOT EXISTS final_score NUMERIC;
                """
            )

            cursor.execute(
                """
                ALTER TABLE allocations
                ADD COLUMN IF NOT EXISTS allocation_time TIMESTAMP NOT NULL DEFAULT NOW();
                """
            )

            cursor.execute(
                """
                ALTER TABLE allocations
                ADD COLUMN IF NOT EXISTS decision_reason TEXT;
                """
            )

            cursor.execute(
                """
                ALTER TABLE allocation_history
                ADD COLUMN IF NOT EXISTS decision_reason TEXT;
                """
            )

        connection.commit()


def replace_oils_data(dataframe: pd.DataFrame) -> int:
    working_df = dataframe.copy()
    if "origin_port" not in working_df.columns and "origin_country" in working_df.columns:
        working_df["origin_port"] = working_df["origin_country"]
    if "origin_country" not in working_df.columns and "origin_port" in working_df.columns:
        working_df["origin_country"] = working_df["origin_port"]

    records = _to_db_records(
        working_df,
        [
            "oil_id",
            "oil_type",
            "delivery_deadline",
            "origin_port",
            "origin_country",
            "destination_port",
            "quantity_mt",
        ],
    )

    query = """
        INSERT INTO oils (
            oil_id, oil_type, delivery_deadline, origin_port, origin_country, destination_port, quantity_mt
        )
        VALUES %s;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE oils CASCADE;")
            execute_values(cursor, query, records)
        connection.commit()

    return len(records)


def replace_ships_data(dataframe: pd.DataFrame) -> int:
    records = _to_db_records(
        dataframe,
        ["ship_id", "capacity_mt", "last_oil_type", "available_date", "available_port"],
    )

    query = """
        INSERT INTO ships (
            ship_id, capacity_mt, last_oil_type, available_date, available_port
        )
        VALUES %s;
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE ships CASCADE;")
            execute_values(cursor, query, records)
        connection.commit()

    return len(records)


def load_oils_and_ships() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with get_connection() as connection:
        oils = pd.read_sql_query(
            """
            SELECT
                oil_id,
                oil_type,
                delivery_deadline,
                COALESCE(origin_port, origin_country) AS origin_port,
                destination_port,
                quantity_mt
            FROM oils
            ORDER BY delivery_deadline ASC;
            """,
            connection,
        )

        ships = pd.read_sql_query(
            """
            SELECT ship_id, capacity_mt, last_oil_type, available_date, available_port
            FROM ships
            ORDER BY available_date ASC;
            """,
            connection,
        )

    return oils, ships


def get_allocated_ship_context() -> pd.DataFrame:
    with get_connection() as connection:
        query = """
            SELECT
                a.oil_id,
                a.ship_id,
                a.status,
                a.final_score,
                COALESCE(o.origin_port, o.origin_country) AS origin_port,
                o.destination_port,
                s.capacity_mt
            FROM allocations a
            INNER JOIN oils o ON o.oil_id = a.oil_id
            INNER JOIN ships s ON s.ship_id = a.ship_id
            WHERE a.status = 'allocated'
            ORDER BY a.id ASC;
        """
        return pd.read_sql_query(query, connection)


def replace_allocations(allocation_rows: List[Dict]) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if allocation_rows:
                run_id = int(datetime.utcnow().timestamp() * 1000)
                cursor.execute("DELETE FROM allocations;")

                records = [
                    (
                        run_id,
                        row["oil_id"],
                        row.get("ship_id"),
                        row["status"],
                        row.get("reason"),
                        row.get("decision_reason", row.get("reason")),
                        row.get("final_score"),
                        row.get("allocation_time"),
                    )
                    for row in allocation_rows
                ]
                execute_values(
                    cursor,
                    """
                    INSERT INTO allocations (run_id, oil_id, ship_id, status, reason, decision_reason, final_score, allocation_time)
                    VALUES %s;
                    """,
                    records,
                )

                execute_values(
                    cursor,
                    """
                    INSERT INTO allocation_history (run_id, oil_id, ship_id, status, reason, decision_reason, final_score, allocation_time)
                    VALUES %s;
                    """,
                    records,
                )
        connection.commit()


def get_overview() -> Dict[str, int]:
    with get_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT COUNT(*)::INT AS total_ships FROM ships;")
            total_ships = cursor.fetchone()["total_ships"]

            cursor.execute("SELECT COALESCE(MAX(run_id), 0) AS latest_run_id FROM allocation_history;")
            latest_run_id = int(cursor.fetchone()["latest_run_id"])

            cursor.execute(
                """
                SELECT COUNT(DISTINCT ship_id)::INT AS allocated_ships
                FROM allocation_history
                WHERE run_id = %s AND status = 'allocated' AND ship_id IS NOT NULL;
                """,
                (latest_run_id,),
            )
            allocated_ships = cursor.fetchone()["allocated_ships"]

            cursor.execute("SELECT COUNT(*)::INT AS total_oil_orders FROM oils;")
            total_orders = cursor.fetchone()["total_oil_orders"]

            cursor.execute(
                """
                SELECT COUNT(*)::INT AS allocated_orders
                FROM allocation_history
                WHERE run_id = %s AND status = 'allocated';
                """,
                (latest_run_id,),
            )
            allocated_orders = cursor.fetchone()["allocated_orders"]

    return {
        "total_ships": total_ships,
        "allocated_ships": allocated_ships,
        "total_orders": total_orders,
        "allocated_orders": allocated_orders,
        "unallocated_orders": max(total_orders - allocated_orders, 0),
    }


def get_recent_allocations(limit: int | None = None) -> pd.DataFrame:
    with get_connection() as connection:
        base_query = """
            SELECT oil_id, ship_id, status,
                   COALESCE(decision_reason, reason) AS decision_reason,
                   final_score,
                   allocation_time,
                   run_id
            FROM allocation_history
            WHERE run_id = (SELECT COALESCE(MAX(run_id), 0) FROM allocation_history)
            ORDER BY id ASC
        """

        if limit is None:
            return pd.read_sql_query(base_query + ";", connection)

        return pd.read_sql_query(base_query + " LIMIT %s;", connection, params=(int(limit),))


def get_previous_run_allocations() -> pd.DataFrame:
    with get_connection() as connection:
        query = """
            WITH latest AS (
                SELECT COALESCE(MAX(run_id), 0) AS latest_run_id
                FROM allocation_history
            ),
            previous AS (
                SELECT COALESCE(MAX(run_id), 0) AS previous_run_id
                FROM allocation_history, latest
                WHERE run_id < latest.latest_run_id
            )
                 SELECT oil_id, ship_id, status,
                     COALESCE(decision_reason, reason) AS decision_reason,
                     final_score,
                   allocation_time,
                   run_id
            FROM allocation_history, previous
            WHERE allocation_history.run_id = previous.previous_run_id
            ORDER BY id ASC;
        """
        return pd.read_sql_query(query, connection)
