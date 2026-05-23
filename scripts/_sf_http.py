"""SolarFlare controller — HTTP request handler."""
import json
import logging
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import parse_qs, urlparse

from _sf_db import (get_all, get_by_id, insert_entry, set_status, now_iso,
                    get_last_user)
from _sf_scheduler import VALID_SLOTS, get_free_slots

log = logging.getLogger(__name__)

VALID_DEVICE_PREFS = {"any", "cuda", "mps", "cpu"}


class Handler(BaseHTTPRequestHandler):
    conn: sqlite3.Connection
    db_lock: threading.Lock

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def send_json(self, code: int, data) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> Optional[dict]:
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n)) if n > 0 else {}
        except Exception:
            return None

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/submit":
            self._submit()
        elif re.match(r"^/cancel/\d+$", path):
            self._cancel(int(path.split("/")[-1]))
        else:
            self.send_json(404, {"error": "not found"})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)
        if path == "/queue":
            self._queue(qs.get("user", [None])[0])
        elif path == "/status":
            self._status()
        else:
            self.send_json(404, {"error": "not found"})

    def _submit(self) -> None:
        data = self.read_json_body()
        if data is None:
            self.send_json(400, {"error": "invalid JSON"}); return
        user = (data.get("user") or "").strip()
        if not user:
            self.send_json(400, {"error": "user required"}); return
        step = (data.get("step_name") or "").strip()
        script = (data.get("script_path") or "").strip()
        if not step or not script:
            self.send_json(400, {"error": "step_name and script_path required"}); return
        dev = data.get("device_pref", "any")
        if dev not in VALID_DEVICE_PREFS:
            self.send_json(400, {"error": f"device_pref must be one of {sorted(VALID_DEVICE_PREFS)}"}); return
        slot = data.get("slot_pref") or None
        if slot and slot not in VALID_SLOTS:
            self.send_json(400, {"error": f"slot_pref must be one of {VALID_SLOTS}"}); return
        with self.db_lock:
            eid = insert_entry(self.conn, data)
        log.info("Queued: id=%d user=%s step=%s", eid, user, step)
        self.send_json(201, {"id": eid, "status": "queued"})

    def _cancel(self, eid: int) -> None:
        with self.db_lock:
            e = get_by_id(self.conn, eid)
            if e is None:
                self.send_json(404, {"error": "not found"}); return
            if e["status"] != "queued":
                self.send_json(409, {"error": f"cannot cancel status={e['status']}"}); return
            set_status(self.conn, eid, "cancelled", {"finished_at": now_iso()})
        log.info("Cancelled id=%d", eid)
        self.send_json(200, {"id": eid, "status": "cancelled"})

    def _queue(self, user: Optional[str]) -> None:
        with self.db_lock:
            rows = get_all(self.conn, user)
        self.send_json(200, [dict(r) for r in rows])

    def _status(self) -> None:
        with self.db_lock:
            running = self.conn.execute(
                "SELECT * FROM queue_entries WHERE status='running'"
            ).fetchall()
            depths = self.conn.execute(
                "SELECT user, COUNT(*) as n FROM queue_entries WHERE status='queued' GROUP BY user"
            ).fetchall()
            last = get_last_user(self.conn)
        self.send_json(200, {
            "running": [dict(r) for r in running],
            "queue_depth": {r["user"]: r["n"] for r in depths},
            "round_robin_pointer": last,
            "free_slots": get_free_slots(),
        })


def make_handler(conn: sqlite3.Connection, db_lock: threading.Lock):
    class BoundHandler(Handler):
        pass
    BoundHandler.conn = conn
    BoundHandler.db_lock = db_lock
    return BoundHandler
