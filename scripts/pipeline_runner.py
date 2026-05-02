"""Unified video processing pipeline — multi-signal decision, non-blocking.

Runs QC + shipping_label/OCR + damage detection sequentially. No module
gates the next; failures become a default-negative signal in the output,
never a hard stop. Always returns a structured JSON result.

Decision layer:
    tracking_valid AND damage_detected         -> AUTO_PASS
    damage_detected AND NOT tracking_valid     -> REVIEW_REQUIRED
    NOT damage_detected                        -> NO_DAMAGE
    else                                       -> INVALID_INPUT (unreachable
        with current matrix; reserved for future use)

CLI:
    python scripts/pipeline_runner.py --input video.mp4
    python scripts/pipeline_runner.py --input video.mp4 --debug
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Sibling-module imports work regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qc_video as qc_mod
import check_tracking as ct_mod
import infer as infer_mod

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "outputs" / "pipeline"

DEFAULT_QC_WEIGHTS = ROOT / "models" / "rl_yolov8n_v1" / "weights" / "best.pt"
DEFAULT_DAMAGE_WEIGHTS = ROOT / "models" / "rl_yolov8n_v2" / "weights" / "best.pt"

# Lowered to 0.1 for debugging (per spec). Real production threshold typically
# lives around 0.25-0.4 once the shipping_label model is trained on enough data.
DEFAULT_SHIPPING_LABEL_CONF = 0.1
DEFAULT_SHIPPING_LABEL_SAMPLE_COUNT = 12


def resolve_shipping_label_weights() -> Path:
    """Pick the latest models/rl_shipping_label_*/weights/best.pt by mtime.

    Returns a non-existent placeholder path if nothing's been trained yet —
    callers handle the missing-weights case explicitly (no_label_detected).
    """
    models_dir = ROOT / "models"
    candidates: list[tuple[float, Path]] = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("rl_shipping_label_"):
                continue
            best = d / "weights" / "best.pt"
            if best.exists():
                candidates.append((d.stat().st_mtime, best))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    # Stable placeholder; doesn't exist on disk → triggers weights_missing branch.
    return models_dir / "rl_shipping_label_<not_trained_yet>" / "weights" / "best.pt"


# Resolved at import time so the rest of the module can reference a stable path.
DEFAULT_TRACKING_WEIGHTS = resolve_shipping_label_weights()


# ---------- per-module wrappers (each returns a structured signal, never raises) ----------

def run_qc(video: Path, weights: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (signal_block, raw_for_debug)."""
    try:
        r = qc_mod.qc_video(video_path=video, weights=weights, save=False)
        signal = {
            "status": r.get("status", "ERROR"),
            "reasons": r.get("reasons", []),
        }
        return signal, r
    except BaseException as exc:  # qc_video raises SystemExit on unreadable input
        signal = {"status": "ERROR", "reasons": [f"exception:{type(exc).__name__}:{exc}"]}
        return signal, {"error": str(exc)}


def run_shipping_label_detection(
    video: Path,
    weights: Path,
    conf_threshold: float = DEFAULT_SHIPPING_LABEL_CONF,
    sample_count: int = DEFAULT_SHIPPING_LABEL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Explicit YOLO shipping_label detection — runs BEFORE OCR.

    Returns a structured detection report. Logs to stdout: model path,
    detection counts, per-frame confidences. Does NOT run OCR.

    Status values:
        weights_missing  — no .pt at the given path
        class_missing    — model loaded but doesn't expose 'shipping_label'
        unreadable       — couldn't sample frames from the video
        no_label_detected — class exists but the model fired on 0 frames
        detected         — at least 1 frame had a box at >= conf_threshold
    """
    print(f"[shipping_label] weights path: {weights}", flush=True)
    print(f"[shipping_label] conf threshold: {conf_threshold}", flush=True)

    if not Path(weights).exists():
        print(f"[shipping_label] ⚠ weights file does NOT exist", flush=True)
        return {
            "status": "weights_missing",
            "weights_path": str(weights),
            "conf_threshold": conf_threshold,
            "frames_sampled": 0,
            "detections_count": 0,
            "confidences": [],
            "best_frame_idx": -1,
            "best_confidence": 0.0,
            "detections": [],
        }

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
    except Exception as exc:
        print(f"[shipping_label] model load failed: {exc}", flush=True)
        return {
            "status": "weights_load_error",
            "weights_path": str(weights),
            "error": f"{type(exc).__name__}: {exc}",
            "conf_threshold": conf_threshold,
            "frames_sampled": 0, "detections_count": 0, "confidences": [],
            "best_frame_idx": -1, "best_confidence": 0.0, "detections": [],
        }

    classes = list(getattr(model, "names", {}).values())
    print(f"[shipping_label] model classes: {classes}", flush=True)

    class_id = ct_mod.resolve_class_id(model, "shipping_label")
    if class_id is None:
        print(f"[shipping_label] ⚠ class 'shipping_label' not in model classes", flush=True)
        return {
            "status": "class_missing",
            "weights_path": str(weights),
            "model_classes": classes,
            "conf_threshold": conf_threshold,
            "frames_sampled": 0, "detections_count": 0, "confidences": [],
            "best_frame_idx": -1, "best_confidence": 0.0, "detections": [],
        }

    frames = ct_mod.sample_frames(Path(video), sample_count)
    if not frames:
        print(f"[shipping_label] ⚠ no frames sampled (video unreadable)", flush=True)
        return {
            "status": "unreadable",
            "weights_path": str(weights),
            "conf_threshold": conf_threshold,
            "frames_sampled": 0, "detections_count": 0, "confidences": [],
            "best_frame_idx": -1, "best_confidence": 0.0, "detections": [],
        }

    boxes_per_frame = ct_mod.detect_label_boxes(model, frames, class_id, conf_threshold)
    detections: list[dict[str, Any]] = []
    confidences: list[float] = []
    best_frame_idx = -1
    best_conf = 0.0
    for i, box in enumerate(boxes_per_frame):
        if box is None:
            detections.append({"frame": i, "detected": False,
                               "confidence": 0.0, "bbox_xyxy": None})
            continue
        x1, y1, x2, y2, c = box
        detections.append({
            "frame": i,
            "detected": True,
            "confidence": round(float(c), 4),
            "bbox_xyxy": [round(float(x1), 1), round(float(y1), 1),
                          round(float(x2), 1), round(float(y2), 1)],
        })
        confidences.append(float(c))
        if c > best_conf:
            best_conf = float(c)
            best_frame_idx = i

    detections_count = len(confidences)
    print(f"[shipping_label] frames sampled: {len(frames)}", flush=True)
    print(f"[shipping_label] detections: {detections_count}/{len(frames)}", flush=True)
    print(f"[shipping_label] confidences: {[round(c, 3) for c in confidences]}", flush=True)
    if detections_count == 0:
        print("[shipping_label] ⚠ NO DETECTION — model didn't fire on any frame", flush=True)

    return {
        "status": "detected" if detections_count > 0 else "no_label_detected",
        "weights_path": str(weights),
        "model_classes": classes,
        "conf_threshold": conf_threshold,
        "frames_sampled": len(frames),
        "detections_count": detections_count,
        "confidences": [round(c, 3) for c in confidences],
        "best_frame_idx": best_frame_idx,
        "best_confidence": round(best_conf, 3),
        "detections": detections,
    }


def run_tracking(video: Path, weights: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tracking signal: detection (BEFORE OCR) → OCR only on detected boxes.

    Spec contract enforced:
      1. Run shipping_label detection on sampled frames
      2. If 0 detections → status=no_label_detected, NO OCR
      3. If ≥1 detection → OCR each crop, validate against tracking pattern
      4. Surface detection details (count, confidences, bboxes) for UI overlay
    """
    # Step A: explicit detection step (logs everything for debugging).
    det = run_shipping_label_detection(
        video=video, weights=weights,
        conf_threshold=DEFAULT_SHIPPING_LABEL_CONF,
        sample_count=DEFAULT_SHIPPING_LABEL_SAMPLE_COUNT,
    )

    # If the detection step short-circuits, return without invoking OCR.
    if det["status"] != "detected":
        signal: dict[str, Any] = {
            "visible": False,
            "text": "",
            "valid": False,
            "status": "no_label_detected",
            "detection": det,  # full report for UI display
            "error": det["status"],
        }
        return signal, {"detection": det, "ocr": None}

    # Step B: OCR only when there are boxes. check_tracking handles the
    # crop+OCR per-frame; strict_no_box_fallback=True skips frames without
    # boxes (defense in depth — we already know there are boxes).
    try:
        r = ct_mod.check_tracking(
            video_path=video,
            weights=weights,
            conf=DEFAULT_SHIPPING_LABEL_CONF,
            save_debug=False,
            strict_no_box_fallback=True,
        )
        status = r.get("status", "FAIL")
        signal = {
            "visible": int(r.get("label_detected_frames", 0)) > 0,
            "text": r.get("best_text", "") or "",
            "valid": status == "PASS",
            "status": status,
            "detection": det,
        }
        if status == "no_label_detected":
            # Edge case: detection saw boxes but check_tracking sampled
            # different frames and missed them. Surface it consistently.
            signal["error"] = "no_label_detected"
        return signal, {"detection": det, "ocr": r}
    except BaseException as exc:
        return {
            "visible": True,  # detection said yes
            "text": "",
            "valid": False,
            "status": "OCR_ERROR",
            "detection": det,
            "error": f"{type(exc).__name__}:{exc}",
        }, {"detection": det, "ocr_error": str(exc)}


def run_damage(
    video: Path, weights: Path, fps: float, conf: float, device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        if not Path(weights).exists():
            signal = {
                "detected": False, "count": 0, "confidence": 0.0,
                "error": f"weights_missing:{weights}",
            }
            return signal, {"error": f"weights_missing:{weights}"}
        from ultralytics import YOLO
        model = YOLO(str(weights))
        r = infer_mod.analyze_file(
            file=video, model=model,
            fps=fps, conf=conf, iou=0.5, imgsz=640, batch=16,
            device=device, save_annotated=False,
        )
        issue_counts = r.get("issue_counts", {}) or {}
        signal = {
            "detected": "damage" in (r.get("issues_detected") or []),
            "count": int(issue_counts.get("damage", 0)),  # frames with damage
            "confidence": float(r.get("confidence", 0.0)),
        }
        return signal, r
    except BaseException as exc:
        signal = {
            "detected": False, "count": 0, "confidence": 0.0,
            "error": f"{type(exc).__name__}:{exc}",
        }
        return signal, {"error": str(exc)}


# ---------- decision layer ----------

def decide(tracking_valid: bool, damage_detected: bool) -> tuple[str, str]:
    """Returns (final_status, confidence_note)."""
    if tracking_valid and damage_detected:
        return "AUTO_PASS", "tracking validated + damage detected"
    if damage_detected and not tracking_valid:
        return "REVIEW_REQUIRED", "damage detected; tracking unreadable — needs human review"
    if not damage_detected:
        return "NO_DAMAGE", (
            "no damage signal" + ("" if tracking_valid else "; tracking also unreadable")
        )
    return "INVALID_INPUT", "no usable signal"


# ---------- orchestrator ----------

def analyze(
    video: Path,
    qc_weights: Path,
    tracking_weights: Path,
    damage_weights: Path,
    fps: float,
    conf: float,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (final_result, debug_blob)."""
    timings: dict[str, float] = {}

    t = time.time()
    qc_signal, qc_raw = run_qc(video, qc_weights)
    timings["qc_sec"] = round(time.time() - t, 3)

    t = time.time()
    tracking_signal, tracking_raw = run_tracking(video, tracking_weights)
    timings["tracking_sec"] = round(time.time() - t, 3)

    t = time.time()
    damage_signal, damage_raw = run_damage(video, damage_weights, fps=fps, conf=conf, device=device)
    timings["damage_sec"] = round(time.time() - t, 3)

    final_status, note = decide(
        tracking_valid=bool(tracking_signal.get("valid")),
        damage_detected=bool(damage_signal.get("detected")),
    )

    # Surface signal-quality caveats in the note even when not decision-gating.
    caveats: list[str] = []
    if qc_signal.get("status") != "PASS":
        caveats.append(f"qc={qc_signal.get('status')}")
    for sec, key in (("qc", qc_signal), ("tracking", tracking_signal), ("damage", damage_signal)):
        if "error" in key:
            caveats.append(f"{sec}_error")
    if caveats:
        note = f"{note} [{', '.join(caveats)}]"

    result = {
        "video": str(video),
        "video_id": qc_mod.derive_video_id(video),
        "qc": qc_signal,
        "tracking": tracking_signal,
        "damage": damage_signal,
        "final_status": final_status,
        "confidence_note": note,
        "timings": timings,
    }
    debug = {"qc_raw": qc_raw, "tracking_raw": tracking_raw, "damage_raw": damage_raw}
    return result, debug


# ---------- CLI ----------

def _print_human(result: dict[str, Any]) -> None:
    color = {
        "AUTO_PASS": "\033[92m",
        "REVIEW_REQUIRED": "\033[93m",
        "NO_DAMAGE": "\033[91m",
        "INVALID_INPUT": "\033[91m",
    }.get(result["final_status"], "")
    reset = "\033[0m"
    qc, tr, dm = result["qc"], result["tracking"], result["damage"]
    print("=" * 72)
    print(f"  Video        : {result['video']}")
    print(f"  video_id     : {result['video_id']}")
    print(f"  Final        : {color}{result['final_status']}{reset}")
    print(f"  Note         : {result['confidence_note']}")
    print("-" * 72)
    print(f"  QC           : {qc['status']:<6} reasons={qc['reasons']}")
    print(f"  Tracking     : visible={tr['visible']}  valid={tr['valid']}  text={tr['text'] or '(none)'!r}")
    print(f"  Damage       : detected={dm['detected']}  count={dm['count']}  conf={dm['confidence']}")
    t = result["timings"]
    print(f"  Timings (s)  : qc={t['qc_sec']}  tracking={t['tracking_sec']}  damage={t['damage_sec']}")
    print("=" * 72)


def run_pipeline(video_path,
                 tracking_weights: Path | None = None) -> dict[str, Any]:
    """Convenience wrapper for programmatic callers (e.g. Streamlit).

    tracking_weights: optional override pointing at a specific
        models/rl_shipping_label_<version>/weights/best.pt. If None, the
        latest such model is auto-resolved at call time (so the pipeline
        always uses the freshest trained shipping_label model).
    """
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    # Re-resolve at call time, not import time, so a model trained mid-session
    # is picked up without restarting Streamlit.
    tw = Path(tracking_weights) if tracking_weights else resolve_shipping_label_weights()
    result, _ = analyze(
        video=video,
        qc_weights=DEFAULT_QC_WEIGHTS,
        tracking_weights=tw,
        damage_weights=DEFAULT_DAMAGE_WEIGHTS,
        fps=2.0, conf=0.35, device="cpu",
    )
    return result


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
    ap = argparse.ArgumentParser(description="Unified non-blocking video pipeline.")
    ap.add_argument("--input", required=True, type=Path, help="Video / image / folder")
    ap.add_argument("--qc-weights", type=Path, default=DEFAULT_QC_WEIGHTS)
    ap.add_argument("--tracking-weights", type=Path, default=DEFAULT_TRACKING_WEIGHTS)
    ap.add_argument("--damage-weights", type=Path, default=DEFAULT_DAMAGE_WEIGHTS)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--debug", action="store_true", help="Print raw per-module outputs")
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

    all_results: list[dict[str, Any]] = []
    for f in iter_inputs(target):
        result, debug = analyze(
            video=f,
            qc_weights=args.qc_weights,
            tracking_weights=args.tracking_weights,
            damage_weights=args.damage_weights,
            fps=args.fps, conf=args.conf, device=args.device,
        )
        _print_human(result)
        if args.debug:
            print("--- debug (raw module outputs) ---")
            print(json.dumps(debug, indent=2, ensure_ascii=False, default=str))

        if not args.no_save:
            (PIPELINE_DIR / f"{result['video_id']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        all_results.append(result)

    print("\n--- JSON ---")
    json.dump(
        all_results if len(all_results) != 1 else all_results[0],
        sys.stdout, indent=2, ensure_ascii=False,
    )
    sys.stdout.write("\n")
    return 0  # never non-zero — non-blocking by contract


if __name__ == "__main__":
    raise SystemExit(main())
