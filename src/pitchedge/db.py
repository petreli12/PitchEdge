"""Single database access layer.

Every other module talks to Postgres through this module: there are no raw
connection strings scattered elsewhere, and all SQL is parameterized (never
string-formatted) to avoid injection and quoting bugs.

Usage:
    from pitchedge import db

    rows = db.fetch_all("SELECT * FROM teams WHERE confederation = :conf",
                        {"conf": "UEFA"})
    db.execute("INSERT INTO teams (team_id, name) VALUES (:id, :name)",
               {"id": 1, "name": "Spain"})
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, Mapping, Sequence

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

from pitchedge import config

# Bound how long a connection attempt blocks (seconds). Keeps the app from
# hanging indefinitely when Postgres is unreachable.
CONNECT_TIMEOUT_S = 10


@lru_cache(maxsize=None)
def get_engine(db_url: str | None = None) -> Engine:
    """Return a process-wide SQLAlchemy Engine (one per distinct URL).

    The engine is a connection-pool factory and is safe to share. ``db_url``
    defaults to ``config.DB_URL``; pass an explicit URL for tests against a
    throwaway database.
    """
    url = db_url or config.DB_URL
    return create_engine(
        url,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_S},
    )


@contextmanager
def connect(db_url: str | None = None) -> Iterator[Connection]:
    """Yield a transactional connection that commits on success, rolls back on
    error, and is always closed.
    """
    engine = get_engine(db_url)
    with engine.begin() as conn:
        yield conn


def execute(
    sql: str,
    params: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    db_url: str | None = None,
) -> int:
    """Run a write statement (INSERT/UPDATE/DELETE/DDL) with bound parameters.

    Pass a list of param mappings to execute the statement once per mapping
    (executemany). Returns the affected row count when available, else -1.
    """
    with connect(db_url) as conn:
        result = conn.execute(text(sql), params or {})
        return result.rowcount if result.rowcount is not None else -1


def execute_script(sql: str, *, db_url: str | None = None) -> None:
    """Run a multi-statement SQL script in a single transaction.

    Used for migrations: the DDL/PL-pgSQL is sent to the driver as one block.
    """
    engine = get_engine(db_url)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(sql)
        raw.commit()
    finally:
        raw.close()


def fetch_all(
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    db_url: str | None = None,
) -> list[dict[str, Any]]:
    """Run a parameterized SELECT and return all rows as dicts."""
    with connect(db_url) as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


def fetch_one(
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Run a parameterized SELECT and return the first row as a dict, or None."""
    with connect(db_url) as conn:
        result = conn.execute(text(sql), params or {})
        row = result.mappings().first()
        return dict(row) if row is not None else None
