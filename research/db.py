from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg
except Exception:
    psycopg = None

DATABASE_URL = os.getenv("DATABASE_URL", "")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/tmp/riskpulse.db")


def _is_postgres() -> bool:
    return DATABASE_URL.startswith("postgres") and psycopg is not None


def init_db() -> None:
    if _is_postgres():
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS source_registry (
                        source_name TEXT PRIMARY KEY, status TEXT NOT NULL,
                        last_checked TIMESTAMPTZ NOT NULL, record_count BIGINT, payload JSONB
                    );
                    CREATE TABLE IF NOT EXISTS model_backtests (
                        id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        model_version TEXT NOT NULL, payload JSONB NOT NULL
                    );
                """)
            conn.commit()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS source_registry (
                source_name TEXT PRIMARY KEY, status TEXT NOT NULL, last_checked TEXT NOT NULL,
                record_count INTEGER, payload TEXT
            );
            CREATE TABLE IF NOT EXISTS model_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                model_version TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)
        conn.commit(); conn.close()


def upsert_source(name: str, status: str, record_count: int | None, payload: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    if _is_postgres():
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO source_registry(source_name,status,last_checked,record_count,payload)
                    VALUES (%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (source_name) DO UPDATE SET status=EXCLUDED.status,last_checked=EXCLUDED.last_checked,
                    record_count=EXCLUDED.record_count,payload=EXCLUDED.payload""",
                    (name, status, now, record_count, json.dumps(payload)))
            conn.commit()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.execute("""INSERT INTO source_registry(source_name,status,last_checked,record_count,payload)
            VALUES (?,?,?,?,?) ON CONFLICT(source_name) DO UPDATE SET status=excluded.status,
            last_checked=excluded.last_checked,record_count=excluded.record_count,payload=excluded.payload""",
            (name, status, now.isoformat(), record_count, json.dumps(payload)))
        conn.commit(); conn.close()


def list_sources() -> list[dict[str, Any]]:
    if _is_postgres():
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT source_name,status,last_checked,record_count,payload FROM source_registry ORDER BY source_name")
                rows = cur.fetchall()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        rows = conn.execute("SELECT source_name,status,last_checked,record_count,payload FROM source_registry ORDER BY source_name").fetchall(); conn.close()
    out=[]
    for name,status,last_checked,count,payload in rows:
        if isinstance(payload, str):
            try: payload=json.loads(payload)
            except Exception: payload={}
        out.append({"source_name":name,"status":status,"last_checked":str(last_checked),"record_count":count,"payload":payload or {}})
    return out


def save_backtest(payload: dict[str, Any], model_version: str="0.1.0") -> None:
    if _is_postgres():
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO model_backtests(model_version,payload) VALUES (%s,%s::jsonb)", (model_version,json.dumps(payload)))
            conn.commit()
    else:
        conn=sqlite3.connect(SQLITE_PATH)
        conn.execute("INSERT INTO model_backtests(created_at,model_version,payload) VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(),model_version,json.dumps(payload)))
        conn.commit(); conn.close()
