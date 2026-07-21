"""SQLite production logging for FrED."""

from __future__ import annotations

import datetime
import logging
import sqlite3
import threading

import pandas as pd

log = logging.getLogger(__name__)


class ProductionDB:
    """Thread-safe SQLite-backed production and session log."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS production_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    part_name   TEXT    NOT NULL,
                    quality     TEXT    NOT NULL,
                    distance    REAL    NOT NULL,
                    x_pixel     INTEGER NOT NULL,
                    y_pixel     INTEGER NOT NULL,
                    z_mm        REAL    NOT NULL,
                    x_robot     REAL,
                    y_robot     REAL,
                    session_id  TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    started_at  TEXT NOT NULL,
                    ended_at    TEXT,
                    total_parts INTEGER DEFAULT 0,
                    approved    INTEGER DEFAULT 0,
                    rejected    INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()
        log.info("Production database ready at '%s'.", self._path)

    def start_session(self) -> str:
        session_id = datetime.datetime.now().strftime("SID-%Y%m%d-%H%M%S")
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
                    (session_id, datetime.datetime.now().isoformat()),
                )
                conn.commit()
        log.info("New production session: %s", session_id)
        return session_id

    def close_session(self, session_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sessions SET ended_at=? WHERE session_id=?",
                    (datetime.datetime.now().isoformat(), session_id),
                )
                conn.commit()

    def log_pick(
        self,
        session_id: str,
        part_name: str,
        quality: str,
        distance: float,
        x_pix: int,
        y_pix: int,
        z_mm: float,
        x_robot: float | None = None,
        y_robot: float | None = None,
    ) -> int:
        """Insert one production event and return its row ID."""
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        approved = 1 if quality == "APPROVED" else 0

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO production_log
                    (timestamp, part_name, quality, distance,
                     x_pixel, y_pixel, z_mm, x_robot, y_robot, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        part_name,
                        quality,
                        distance,
                        x_pix,
                        y_pix,
                        z_mm,
                        x_robot,
                        y_robot,
                        session_id,
                    ),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET total_parts = total_parts + 1,
                        approved    = approved + ?,
                        rejected    = rejected + ?
                    WHERE session_id = ?
                    """,
                    (approved, 1 - approved, session_id),
                )
                conn.commit()
                row_id = int(cursor.lastrowid)

        log.info(
            "DB log #%d — %s [%s] dist=%.1f",
            row_id,
            part_name,
            quality,
            distance,
        )
        return row_id

    def get_recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM production_log ORDER BY id DESC LIMIT ?", (n,)
                ).fetchall()
        return [dict(row) for row in rows]

    def get_session_stats(self, session_id: str) -> dict:
        with self._lock:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
        return dict(row) if row else {}

    def get_totals(self) -> dict:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN quality='APPROVED' THEN 1 ELSE 0 END) AS approved,
                           SUM(CASE WHEN quality='REJECTED' THEN 1 ELSE 0 END) AS rejected
                    FROM production_log
                    """
                ).fetchone()
        return {
            "total": row[0] or 0,
            "approved": row[1] or 0,
            "rejected": row[2] or 0,
        }

    def export_csv(self, path: str) -> None:
        with self._lock:
            with self._connect() as conn:
                dataframe = pd.read_sql_query(
                    "SELECT * FROM production_log ORDER BY id", conn
                )
        dataframe.to_csv(path, index=False)
        log.info("Exported %d rows to '%s'.", len(dataframe), path)
