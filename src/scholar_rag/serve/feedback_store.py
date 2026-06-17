"""Durable storage for answers and the feedback users give them.

This is the point of the whole feedback feature: every answer we generate and
every thumbs up/down lands here as a row.

Backed by SQLite: one file on disk, no server, in the standard library. The
method signatures below are deliberately database-agnostic, so moving to
Postgres (Supabase) or libSQL (Turso) at deploy time means rewriting the SQL
inside this one class — nothing else in the app changes.

Threading note: FastAPI runs sync endpoints in a threadpool, so several threads
may call this store at once. We open a short-lived connection per operation
(SQLite handles this well, especially in WAL mode) rather than sharing one
connection across threads.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL            -- JSON array of the cited passages
);
CREATE TABLE IF NOT EXISTS feedback (
    answer_id   TEXT PRIMARY KEY,        -- one vote per answer; re-voting upserts
    created_at  TEXT NOT NULL,
    vote        TEXT NOT NULL CHECK (vote IN ('up', 'down')),
    reasons     TEXT NOT NULL,           -- JSON array, e.g. ["Wrong","Bad citation"]
    comment     TEXT,
    FOREIGN KEY (answer_id) REFERENCES answers (id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode = WAL")  # concurrent reads while writing
        conn.execute("PRAGMA foreign_keys = ON")   # enforce answer_id must exist
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_answer(self, question: str, answer: str, sources: list[dict[str, Any]]) -> str:
        """Persist a generated answer and return its new id."""
        answer_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO answers (id, created_at, question, answer, sources) "
                "VALUES (?, ?, ?, ?, ?)",
                (answer_id, _now(), question, answer, json.dumps(sources)),
            )
        return answer_id

    def record_feedback(
        self,
        answer_id: str,
        vote: str,
        reasons: list[str],
        comment: str | None,
    ) -> bool:
        """Attach (or replace) a vote on an answer. Returns False if no such answer."""
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM answers WHERE id = ?", (answer_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO feedback (answer_id, created_at, vote, reasons, comment) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(answer_id) DO UPDATE SET "
                "created_at = excluded.created_at, vote = excluded.vote, "
                "reasons = excluded.reasons, comment = excluded.comment",
                (answer_id, _now(), vote, json.dumps(reasons), comment),
            )
        return True

    def delete_feedback(self, answer_id: str) -> None:
        """Remove a vote (the UI's 'undo')."""
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback WHERE answer_id = ?", (answer_id,))