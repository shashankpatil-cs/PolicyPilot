"""Create the local Postgres role, database and pgvector extension from .env.

    cd backend && python scripts/setup_db.py

Configured for local Postgres development. Safe to re-run.
"""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import dotenv_values
from psycopg2 import sql

env = {**dotenv_values(Path(__file__).resolve().parents[1] / ".env"), **os.environ}

db_host = env.get("DB_HOST", "localhost")
db_port = int(env.get("DB_PORT", 5432))
db_name = env.get("DB_NAME", "rbi_intelligence")
db_user = env.get("DB_USER", "postgres")
db_password = env.get("DB_PASSWORD", "postgres")

# Default superuser to db_user / db_password if not explicitly specified
pg_superuser = env.get("PG_SUPERUSER", db_user)
pg_superuser_password = env.get("PG_SUPERUSER_PASSWORD", db_password)


def connect(dbname: str):
    # Explicit connection (not `with conn:`) so autocommit really applies:
    # CREATE DATABASE cannot run inside a transaction block.
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=pg_superuser,
        password=pg_superuser_password,
        dbname=dbname,
    )
    conn.autocommit = True
    return conn


admin_conn = connect("postgres")
try:
    with admin_conn.cursor() as cur:
        if db_user != pg_superuser:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (db_user,))
            verb = "ALTER" if cur.fetchone() else "CREATE"
            cur.execute(
                sql.SQL(f"{verb} ROLE {{}} WITH LOGIN PASSWORD {{}}").format(
                    sql.Identifier(db_user), sql.Literal(db_password)
                )
            )
            print(f"Role '{db_user}' {'updated' if verb == 'ALTER' else 'created'}")

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if cur.fetchone():
            print(f"Database '{db_name}' already exists")
        else:
            cur.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(db_name), sql.Identifier(db_user)
                )
            )
            print(f"Database '{db_name}' created")
finally:
    admin_conn.close()

db_conn = connect(env["DB_NAME"])
try:
    with db_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        print(f"pgvector extension ready (version {cur.fetchone()[0]})")
finally:
    db_conn.close()
