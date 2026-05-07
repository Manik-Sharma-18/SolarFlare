"""SolarFlare controller — SQLite database operations."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute("PRAGMA table_info(queue_entries)").fetchall()
    existing_cols = {row[1] for row in existing}
    if existing and "launched_slot" not in existing_cols:
        conn.execute("ALTER TABLE queue_entries ADD COLUMN launched_slot TEXT")
        conn.commit()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queue_entries (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user          TEXT    NOT NULL,
            step_name     TEXT    NOT NULL,
            script_path   TEXT    NOT NULL,
            args          TEXT    NOT NULL DEFAULT '',
            device_pref   TEXT    NOT NULL DEFAULT 'any',
            slot_pref     TEXT,
            priority      INTEGER NOT NULL DEFAULT 0,
            status        TEXT    NOT NULL DEFAULT 'queued',
            submitted_at  TEXT    NOT NULL,
            launched_at   TEXT,
            launched_slot TEXT,
            finished_at   TEXT,
            result_path   TEXT
        );
        CREATE TABLE IF NOT EXISTS round_robin (
            id        INTEGER PRIMARY KEY CHECK (id = 1),
            last_user TEXT    NOT NULL DEFAULT ''
        );
        INSERT OR IGNORE INTO round_robin (id, last_user) VALUES (1, '');
    """)
    conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_last_user(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT last_user FROM round_robin WHERE id=1").fetchone()
    return row["last_user"] if row else ""


def set_last_user(conn: sqlite3.Connection, user: str) -> None:
    conn.execute("UPDATE round_robin SET last_user=? WHERE id=1", (user,))
    conn.commit()


def insert_entry(conn: sqlite3.Connection, data: dict) -> int:
    cur = conn.execute(
        """INSERT INTO queue_entries
           (user, step_name, script_path, args, device_pref, slot_pref, priority, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (data["user"], data["step_name"], data["script_path"],
         data.get("args", ""), data.get("device_pref", "any"),
         data.get("slot_pref") or None, int(data.get("priority", 0)), now_iso()),
    )
    conn.commit()
    return cur.lastrowid


def get_queued(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM queue_entries WHERE status='queued' ORDER BY priority DESC, submitted_at ASC"
    ).fetchall()


def get_all(conn: sqlite3.Connection, user: Optional[str] = None) -> List[sqlite3.Row]:
    if user:
        return conn.execute(
            "SELECT * FROM queue_entries WHERE user=? ORDER BY priority DESC, submitted_at ASC", (user,)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM queue_entries ORDER BY priority DESC, submitted_at ASC"
    ).fetchall()


def get_by_id(conn: sqlite3.Connection, entry_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM queue_entries WHERE id=?", (entry_id,)).fetchone()


def set_status(conn: sqlite3.Connection, entry_id: int, status: str,
               extra: Optional[dict] = None) -> None:
    if extra:
        fields = ", ".join(f"{k}=?" for k in extra)
        vals = list(extra.values()) + [status, entry_id]
        conn.execute(f"UPDATE queue_entries SET {fields}, status=? WHERE id=?", vals)
    else:
        conn.execute("UPDATE queue_entries SET status=? WHERE id=?", (status, entry_id))
    conn.commit()
