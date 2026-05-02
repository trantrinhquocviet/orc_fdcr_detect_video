"""Long-running stdin/stdout JSON-RPC server for Electron integration.

Why this exists: spawning `python infer.py ...` for every video re-imports
torch/ultralytics every time (~3-5s cold start). Instead, an Electron app
spawns this once, then writes one JSON request per line and reads one JSON
response per line.

Protocol (line-delimited JSON):
    Request:  {"id": "abc", "input": "/path/to/file.mp4", "fps": 2.0,
               "conf": 0.35, "device": "cpu", "save_annotated": false}
    Response: {"id": "abc", "ok": true, "result": { ...same shape as infer.py JSON... }}
              {"id": "abc", "ok": false, "error": "..."}

Start:
    python scripts/serve.py --weights models/rl_yolov8n/weights/best.pt --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

from infer import analyze_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.weights)
    # Warm up so the first real request isn't slow.
    model.predict(source=None, verbose=False) if False else None

    sys.stdout.write(json.dumps({"event": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad_json: {exc}"}) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        try:
            result = analyze_file(
                file=Path(req["input"]).expanduser().resolve(),
                model=model,
                fps=float(req.get("fps", 2.0)),
                conf=float(req.get("conf", 0.35)),
                iou=float(req.get("iou", 0.5)),
                imgsz=int(req.get("imgsz", args.imgsz)),
                batch=int(req.get("batch", args.batch)),
                device=req.get("device", args.device),
                save_annotated=bool(req.get("save_annotated", False)),
            )
            sys.stdout.write(json.dumps({"id": req_id, "ok": True, "result": result}) + "\n")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"id": req_id, "ok": False, "error": str(exc)}) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
