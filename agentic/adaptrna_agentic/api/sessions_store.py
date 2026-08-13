"""Sessions as a managed resource: list with recency, rename, and the delete seam.

Sessions are LangGraph threads in the checkpointer the terminal also uses, so this module
owns no schema of its own — every query here runs against the tables LangGraph created.
That is deliberate: a sidecar table of session metadata would be a second source of truth
that the terminal front end never writes, and a session renamed in the browser would still
show its old name in `chat --list-sessions` (plans/PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md
§2).

Deletion is *not* here — it lives in `toolhub.prune.delete_session`, which the CLI's
`prune sessions` has always used. One deletion path, one set of tests.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
import sqlite3

#: The tables a thread's rows live in. Which of them exist varies with the checkpointer
#: version, so every write walks this list defensively — the same approach, for the same
#: reason, as `prune.delete_session`.
THREAD_TABLES = ("checkpoints", "writes", "checkpoint_blobs")

#: UUIDv6 counts 100-nanosecond intervals from the Gregorian epoch.
_GREGORIAN_EPOCH = datetime(1582, 10, 15, tzinfo=timezone.utc)


def checkpoint_time(checkpoint_id: str) -> Optional[str]:
    """The wall-clock time embedded in a UUIDv6 checkpoint id, as an ISO-8601 string.

    LangGraph mints checkpoint ids as UUIDv6, whose leading 60 bits are a timestamp — so
    `MAX(checkpoint_id)` per thread is genuinely the newest checkpoint, and its time can be
    read without deserializing a single BLOB. Returns None for anything that is not a
    UUIDv6, so a future checkpointer change degrades to "no timestamp" rather than a 500.
    """
    digits = (checkpoint_id or "").replace("-", "")
    if len(digits) != 32 or digits[12] != "6":
        return None

    try:
        ticks = int(digits[0:12] + digits[13:16], 16)
    except ValueError:
        return None

    moment = _GREGORIAN_EPOCH + timedelta(microseconds=ticks / 10)
    return moment.isoformat().replace("+00:00", "Z")


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    # The terminal may hold the same file; match the checkpointer's own settings so a
    # concurrent write waits rather than failing outright (api/deps.py::open_checkpointer).
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def list_sessions(db_path: Path) -> List[dict]:
    """Every session, newest first: `{id, updated_at, checkpoints}`."""
    if not Path(db_path).exists():
        return []

    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT thread_id, COUNT(*), MAX(checkpoint_id) "
            "FROM checkpoints GROUP BY thread_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()

    sessions = [
        {"id": thread_id, "checkpoints": count, "updated_at": checkpoint_time(newest)}
        for thread_id, count, newest in rows
    ]

    # Nulls last, so a session whose id we cannot date does not sort to the top.
    sessions.sort(key=lambda s: (s["updated_at"] is not None, s["updated_at"] or ""),
                  reverse=True)
    return sessions


def session_exists(db_path: Path, thread_id: str) -> bool:
    if not Path(db_path).exists():
        return False

    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1", (thread_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()

    return row is not None


def get_session(db_path: Path, thread_id: str) -> Optional[dict]:
    for session in list_sessions(db_path):
        if session["id"] == thread_id:
            return session
    return None


def rename_session(db_path: Path, old: str, new: str) -> None:
    """Move every row of a thread to a new id, in one transaction.

    `thread_id` is part of the primary key on both tables, so this is an UPDATE of the key
    rather than a metadata edit — which is the price of keeping one name visible to both
    front ends. Callers validate that `new` is free *before* calling; the UNIQUE failure
    here is the backstop, not the check.
    """
    connection = _connect(db_path)
    try:
        with connection:  # commits, or rolls back every table together
            for table in THREAD_TABLES:
                try:
                    connection.execute(
                        f"UPDATE {table} SET thread_id = ? WHERE thread_id = ?", (new, old)
                    )
                except sqlite3.OperationalError:
                    continue  # this checkpointer version does not have that table
    finally:
        connection.close()
