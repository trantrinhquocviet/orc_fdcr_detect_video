"""Detached wrapper: runs split.py + train.py, records status in a JSON state
file. Spawned by review_app's Train tab via subprocess.Popen so the parent
Streamlit process never blocks.

State file lifecycle:
    {status: "running", step: "split"}    — at start
    {status: "running", step: "train"}    — after split succeeds
    {status: "done"|"failed", exit_code, ended_at}  — at end

The Streamlit UI polls this file (and the log file's mtime) to know when to
flip the button between idle / running / done / failed.

Usage (invoked by Streamlit, not directly):
    python scripts/_train_runner.py --data <yaml> --name <run> \
        --epochs 30 --imgsz 640 --batch 4 --device cpu \
        --state-file <path>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_state(state_file: Path, payload: dict) -> None:
    tmp = state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(state_file)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state-file", required=True, type=Path)
    args = ap.parse_args()

    state_file: Path = args.state_file
    state_file.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    pid = os.getpid()

    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = Path(sys.executable)  # fall back to whatever is launching us

    common = {
        "started_at": started_at,
        "pid": pid,
        "command": f"{args.name} ({args.data})",
    }

    print(f"[{_ts()}] === split.py ===", flush=True)
    _write_state(state_file, {**common, "status": "running", "step": "split"})
    r = subprocess.run(
        [str(venv_py), str(ROOT / "scripts" / "split.py"), "--data", args.data],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        _write_state(state_file, {
            **common, "status": "failed", "step": "split",
            "exit_code": r.returncode, "ended_at": time.time(),
        })
        print(f"[{_ts()}] split.py FAILED with exit code {r.returncode}", flush=True)
        return r.returncode

    print(f"[{_ts()}] === train.py ===", flush=True)
    _write_state(state_file, {**common, "status": "running", "step": "train"})
    r = subprocess.run(
        [
            str(venv_py), str(ROOT / "scripts" / "train.py"),
            "--data", args.data,
            "--name", args.name,
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--device", args.device,
        ],
        cwd=str(ROOT),
    )

    final = {
        **common,
        "status": "done" if r.returncode == 0 else "failed",
        "step": "train",
        "exit_code": r.returncode,
        "ended_at": time.time(),
    }
    _write_state(state_file, final)
    print(f"[{_ts()}] === DONE === exit code {r.returncode}", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
