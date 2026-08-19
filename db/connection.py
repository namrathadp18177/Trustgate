"""
Thin connection layer around psycopg2 + pgvector.
Centralizes connection config so the rest of the codebase never touches
env vars directly.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432"),
    "dbname": os.environ.get("POSTGRES_DB", "trustgate"),
    "user": os.environ.get("POSTGRES_USER", "trustgate"),
    "password": os.environ.get("POSTGRES_PASSWORD", "trustgate"),
}


@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(dict_cursor: bool = True):
    with get_connection() as conn:
        cursor_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
        finally:
            cur.close()


def init_schema(schema_path: str = None):
    schema_path = schema_path or os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        ddl = f.read()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(ddl)
        cur.close()
    print("[db] schema initialized")


if __name__ == "__main__":
    init_schema()
