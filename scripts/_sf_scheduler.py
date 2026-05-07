"""SolarFlare controller — slot polling and scheduling logic."""
import logging
import os
import shlex
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

from _sf_db import (get_last_user, get_queued, set_last_user, set_status, now_iso)

log = logging.getLogger(__name__)

VALID_SLOTS = ["mini_mps", "mini_cpu", "studio_mps", "studio_cpu", "5060ti_cuda"]
SLOT_DEVICE: Dict[str, str] = {
    "mini_mps": "mps", "mini_cpu": "cpu",
    "studio_mps": "mps", "studio_cpu": "cpu",
    "5060ti_cuda": "cuda",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "launch_slot.sh"
STATUS_SCRIPT = REPO_ROOT / "scripts" / "slot_status.sh"


# --- Slot status ------------------------------------------------------------

def get_free_slots() -> List[str]:
    try:
        r = subprocess.run(["bash", str(STATUS_SCRIPT)], capture_output=True,
                           text=True, timeout=30)
        return _parse_free(r.stdout)
    except subprocess.TimeoutExpired:
        log.warning("slot_status.sh timed out"); return []
    except Exception as exc:
        log.warning("slot_status.sh error: %s", exc); return []


def _parse_free(output: str) -> List[str]:
    free = []
    for line in output.splitlines():
        line = line.strip()
        for slot in VALID_SLOTS:
            if line.startswith(slot) and "[FREE]" in line:
                free.append(slot); break
    return free


# --- Scheduling helpers -----------------------------------------------------

def _device_ok(pref: str, slot: str) -> bool:
    return pref == "any" or SLOT_DEVICE.get(slot) == pref


def _distinct_users(entries: List[sqlite3.Row]) -> List[str]:
    seen: dict = {}
    for e in entries:
        seen[e["user"]] = True
    return list(seen.keys())


def _rr_order(users: List[str], last: str) -> List[str]:
    if not users: return []
    if last not in users: return users[:]
    i = users.index(last)
    s = (i + 1) % len(users)
    return users[s:] + users[:s]


def _pick_reserved(queued: List[sqlite3.Row], slot: str, last: str) -> Optional[sqlite3.Row]:
    cands = [e for e in queued if e["slot_pref"] == slot]
    if not cands: return None
    for user in _rr_order(_distinct_users(cands), last):
        hits = [e for e in cands if e["user"] == user]
        if hits: return hits[0]
    return None


def _pick_any(queued: List[sqlite3.Row], slot: str, last: str) -> Optional[sqlite3.Row]:
    compat = [e for e in queued if e["slot_pref"] is None and _device_ok(e["device_pref"], slot)]
    if not compat: return None
    for user in _rr_order(_distinct_users(compat), last):
        hits = [e for e in compat if e["user"] == user]
        if hits: return hits[0]
    return None


# --- Launch -----------------------------------------------------------------

def launch_entry(conn: sqlite3.Connection, entry: sqlite3.Row, slot: str) -> bool:
    cmd = ["bash", str(LAUNCH_SCRIPT), slot, entry["script_path"]]
    if entry["args"]:
        cmd.extend(shlex.split(entry["args"]))
    env = os.environ.copy()
    env["SF_USER"] = entry["user"]
    log.info("Launching id=%d user=%s slot=%s script=%s",
             entry["id"], entry["user"], slot, entry["script_path"])
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            log.info("Launched OK: id=%d slot=%s", entry["id"], slot)
            set_status(conn, entry["id"], "running",
                       {"launched_at": now_iso(), "launched_slot": slot})
            return True
        elif r.returncode == 2:
            log.warning("Slot occupied (exit 2): slot=%s id=%d", slot, entry["id"])
        else:
            log.error("launch_slot.sh exit %d for id=%d: %s",
                      r.returncode, entry["id"], r.stderr.strip())
            set_status(conn, entry["id"], "failed", {"finished_at": now_iso()})
    except subprocess.TimeoutExpired:
        log.error("launch_slot.sh timeout id=%d slot=%s", entry["id"], slot)
    except Exception as exc:
        log.error("launch_slot.sh exception id=%d: %s", entry["id"], exc)
        set_status(conn, entry["id"], "failed", {"finished_at": now_iso()})
    return False


def reconcile_running(conn: sqlite3.Connection, free_slots: List[str]) -> None:
    running = conn.execute("SELECT * FROM queue_entries WHERE status='running'").fetchall()
    for e in running:
        slot = e["launched_slot"] or e["slot_pref"]
        if slot and slot in free_slots:
            log.info("Reconcile: id=%d slot=%s → done", e["id"], slot)
            set_status(conn, e["id"], "done", {"finished_at": now_iso()})


# --- Tick -------------------------------------------------------------------

def scheduler_tick(conn: sqlite3.Connection, db_lock: threading.Lock) -> None:
    with db_lock:
        free = get_free_slots()
        if not free: return
        queued = get_queued(conn)
        last = get_last_user(conn)
        reconcile_running(conn, free)
        queued = get_queued(conn)

        reserved = {s for s in free if any(e["slot_pref"] == s for e in queued)}
        dispatched = None
        for slot in free:
            if slot in reserved:
                entry = _pick_reserved(queued, slot, last)
            else:
                entry = _pick_any(queued, slot, last)
            if entry and launch_entry(conn, entry, slot):
                dispatched = entry["user"]
                last = entry["user"]
                queued = [e for e in queued if e["id"] != entry["id"]]

        if dispatched:
            set_last_user(conn, dispatched)


def scheduler_loop(conn: sqlite3.Connection, db_lock: threading.Lock,
                   tick_s: int, stop: threading.Event) -> None:
    log.info("Scheduler started (tick=%ds)", tick_s)
    while not stop.is_set():
        try:
            scheduler_tick(conn, db_lock)
        except Exception as exc:
            log.exception("Tick error: %s", exc)
        stop.wait(tick_s)
    log.info("Scheduler stopped")
