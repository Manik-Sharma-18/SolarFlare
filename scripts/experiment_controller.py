#!/usr/bin/env python3
"""
experiment_controller.py — SolarFlare centralized experiment queue.

HTTP server on 0.0.0.0:7434.  SQLite at .controller/queue.db.
Scheduler ticks every 30s, dispatches via scripts/launch_slot.sh.

Usage:
    python3 scripts/experiment_controller.py [--port 7434] [--tick 30]
"""
# See submodules: _sf_db.py, _sf_scheduler.py, _sf_http.py

import argparse
import logging
import signal
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

# Add scripts/ dir to path so submodules resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sf_db import open_db, init_schema
from _sf_scheduler import scheduler_loop
from _sf_http import make_handler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB  = REPO_ROOT / ".controller" / "queue.db"
DEFAULT_LOG = REPO_ROOT / "logs" / "experiment_controller.log"
DEFAULT_PORT = 7434
DEFAULT_TICK = 30


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)],
    )


def _serve_forever(server: HTTPServer, stop: threading.Event) -> None:
    while not stop.is_set():
        server.handle_request()


def main() -> None:
    p = argparse.ArgumentParser(description="SolarFlare experiment controller")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--tick", type=int, default=DEFAULT_TICK)
    p.add_argument("--db",  type=Path, default=DEFAULT_DB)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = p.parse_args()

    setup_logging(args.log)
    log = logging.getLogger(__name__)
    log.info("Starting SolarFlare controller (port=%d tick=%ds db=%s)",
             args.port, args.tick, args.db)

    conn = open_db(args.db)
    init_schema(conn)
    db_lock = threading.Lock()
    stop = threading.Event()

    def shutdown(sig, _frame):
        log.info("Signal %d — shutting down", sig)
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    sched = threading.Thread(target=scheduler_loop,
                             args=(conn, db_lock, args.tick, stop),
                             daemon=True, name="scheduler")
    sched.start()

    handler_cls = make_handler(conn, db_lock)
    server = HTTPServer(("0.0.0.0", args.port), handler_cls)
    server.timeout = 1.0

    http = threading.Thread(target=_serve_forever, args=(server, stop),
                            daemon=True, name="http")
    http.start()

    log.info("Controller ready on port %d", args.port)
    stop.wait()
    log.info("Stopping")
    server.server_close()
    sched.join(timeout=5)
    log.info("Exited cleanly")


if __name__ == "__main__":
    main()
