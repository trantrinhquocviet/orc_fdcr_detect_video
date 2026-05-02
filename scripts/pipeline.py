"""Non-blocking signal-aggregation pipeline.

Runs all three modules unconditionally and combines their signals into a
single decision. No module gates the next; a module failure becomes a
default-negative signal in the output, never a hard stop.

Modules (always all three):
    qc_video.qc_video            -> qc_status, qc_reasons, qc_metrics
    check_tracking.check_tracking -> tracking_visible, tracking_text
    infer.analyze_file           -> damage_detected (+ raw verdict/issues)

Decision (tracking_visible x damage_detected):
    True  + True   -> PASS
    False + True   -> REVIEW_REQUIRED
    True  + False  -> REJECT
    False + False  -> INVALID_INPUT

QC status is surfaced but does not gate the decision.

CLI:
    python scripts/pipeline.py --input video.mp4
    python scripts/pipeline.py --input folder/ --damage-weights models/rl_yolov8n_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure sibling-module imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qc_video as qc_mod
import check_tracking as ct_mod
import infer as infer_mod

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "outputs" / "pipeline"

DEFAULT_DAMAGE_WEIGHTS = ROOT / "models" / "rl_yolov8n_v2" / "weights" / "best.pt"
DEFAULT_TRACKING_WEIGHTS = ROOT / "models" / "rl_yolov8n_v2" / "weights" / "best.pt"
DEFAULT_QC_WEIGHTS = ROOT / "models" / "rl_yolov8n_v1" / "weights" / "best.pt"


# ---------- per-module wrappers (each returns a structured signal, never raises) ----------

def _empty_qc(error: str) -> dict[str, Any]:
    return {"qc_status": "ERROR", "qc_reasons": [error], "qc_metrics": {}}


def _empty_tracking(error: str) -> dict[str, Any]:
    return {
        "tracking_visible": False,
        "tracking_text": "",
        "tracking_label_frames": 0,
        "tracking_valid_ocr_frames": 0,
        "tracking_error": error,
    }


def _empty_damage(error: str) -> dict[str, Any]:
    return {
        "damage_detected": False,
        "damage_verdict": "error",
        "damage_issues": [],
        "damage_error": error,
    }


def run_qc(video: Path, weights: Path) -> dict[str, Any]:
    try:
        r = qc_mod.qc_video(video_path=video, weights=weights, save=False)
    except BaseException as exc:  # qc_video raises SystemExit on unreadable input
        return _empty_qc(str(exc) or type(exc).__name__)
    return {
        "qc_status": r.get("status", "ERROR"),
        "qc_reasons": r.get("reasons", []),
        "qc_metrics": r.get("metrics", {}),
    }


def run_tracking(video: Path, weights: Path) -> dict[str, Any]:
    try:
        r = ct_mod.check_tracking(
            video_path=video,
            weights=weights,
            save_debug=False,
        )
    except BaseException as exc:
        return _empty_tracking(str(exc) or type(exc).__name__)
    return {
        "tracking_visible": r.get("status") == "PASS",
        "tracking_text": r.get("best_text", "") or "",
        "tracking_label_frames": int(r.get("label_detected_frames", 0)),
        "tracking_valid_ocr_frames": int(r.get("valid_ocr_frames", 0)),
        "tracking_decision_rule": r.get("decision_rule", ""),
    }


def run_damage(video: Path, weights: Path, fps: float, conf: float, device: str) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
        if not Path(weights).exists():
            return _empty_damage(f"weights_missing: {weights}")
        model = YOLO(str(weights))
        r = infer_mod.analyze_file(
            file=video, model=model,
            fps=fps, conf=conf, iou=0.5, imgsz=640, batch=16,
            device=device, save_annotated=False,
        )
    except BaseException as exc:
        return _empty_damage(str(exc) or type(exc).__name__)
    issues = r.get("issues_detected", []) or []
    return {
        "damage_detected": "damage" in issues,
        "damage_verdict": r.get("verdict", "ok"),
        "damage_issues": issues,
        "damage_frame_count": r.get("frame_count", 0),
        "damage_confidence": r.get("confidence", 0.0),
    }


# ---------- decision layer ----------

def decide(tracking_visible: bool, damage_detected: bool) -> str:
    if tracking_visible and damage_detected:
        return "PASS"
    if not tracking_visible and damage_detected:
        return "REVIEW_REQUIRED"
    if tracking_visible and not damage_detected:
        return "REJECT"
    return "INVALID_INPUT"


# ---------- orchestrator ----------

def analyze(
    video: Path,
    qc_weights: Path,
    tracking_weights: Path,
    damage_weights: Path,
    fps: float,
    conf: float,
    device: str,
) -> dict[str, Any]:
    t0 = time.time()
    qc = run_qc(video, qc_weights)
    tracking = run_tracking(video, tracking_weights)
    damage = run_damage(video, damage_weights, fps=fps, conf=conf, device=device)

    decision = decide(
        tracking_visible=bool(tracking["tracking_visible"]),
        damage_detected=bool(damage["damage_detected"]),
    )

    return {
        "video": str(video),
        "video_id": qc_mod.derive_video_id(video),
        "decision": decision,
        **qc,
        **tracking,
        **damage,
        "elapsed_sec": round(time.time() - t0, 3),
    }


def iter_inputs(target: Path):
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.suffix.lower() in infer_mod.VIDEO_EXTS | infer_mod.IMG_EXTS:
                yield p
        return
    raise FileNotFoundError(target)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Non-blocking signal-aggregation pipeline.")
    ap.add_argument("--input", required=True, type=Path, help="Video / image / folder")
    ap.add_argument("--qc-weights", type=Path, default=DEFAULT_QC_WEIGHTS)
    ap.add_argument("--tracking-weights", type=Path, default=DEFAULT_TRACKING_WEIGHTS)
    ap.add_argument("--damage-weights", type=Path, default=DEFAULT_DAMAGE_WEIGHTS)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-save", action="store_true",
                    help="Skip writing outputs/pipeline/<video_id>.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    target = args.input.expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Not found: {target}")

    if not args.no_save:
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for f in iter_inputs(target):
        result = analyze(
            video=f,
            qc_weights=args.qc_weights,
            tracking_weights=args.tracking_weights,
            damage_weights=args.damage_weights,
            fps=args.fps,
            conf=args.conf,
            device=args.device,
        )
        if not args.no_save:
            (PIPELINE_DIR / f"{result['video_id']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        results.append(result)

    json.dump(
        results if len(results) != 1 else results[0],
        sys.stdout, indent=2, ensure_ascii=False,
    )
    sys.stdout.write("\n")
    return 0  # never non-zero — non-blocking by contract


if __name__ == "__main__":
    raise SystemExit(main())
