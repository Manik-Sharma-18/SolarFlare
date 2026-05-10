"""JSONL run logger — append-only per-epoch records, decorator-based.
Wraps train_one_epoch / validate at the call site so trainer stays pure;
one open+write+close per epoch is ~ms vs minutes of compute."""
from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import torch


def _scalarize(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item()) if v.numel() == 1 else v.shape
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def log_jsonl(path: str | Path, kind: str) -> Callable:
    """Decorator: capture wrapped fn's return dict + timing, append one JSONL line."""
    path = Path(path)

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            rec: dict[str, Any] = {
                "kind": kind,
                "ts": time.time(),
                "elapsed_s": round(time.time() - t0, 2),
            }
            if isinstance(result, dict):
                rec.update({k: _scalarize(v) for k, v in result.items()})
            # Sniff TrainState in args for epoch / global_step.
            for a in args:
                if hasattr(a, "epoch") and hasattr(a, "global_step"):
                    rec["epoch"] = int(a.epoch)
                    rec["global_step"] = int(a.global_step)
                    break
            _append(path, rec)
            return result
        return wrapped
    return deco


def log_meta(path: str | Path, **kw) -> None:
    """One-shot record (run start, config snapshot, ckpt path, etc.)."""
    rec = {"kind": "meta", "ts": time.time(), **kw}
    _append(Path(path), rec)
