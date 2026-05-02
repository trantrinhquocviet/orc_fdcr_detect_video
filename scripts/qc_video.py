"""Rule-based video quality control — gate videos before frame extraction.

Rules:
    (A) duration   — total seconds < min_duration                     [CRITICAL]
    (B) motion     — almost no frame-to-frame difference              [CRITICAL]
    (C) signal     — 0 YOLO detections across sampled frames          [CRITICAL]
    (D) blur       — most sampled frames have low Laplacian variance  [CRITICAL]
    (E) brightness — mean intensity < threshold                       (warning)
    (F) occlusion  — detected boxes cover very little frame area      (warning)
    (G) framing    — largest detected box still too small             (warning)

Output:
    dict with status (PASS|FAIL), reasons[], metrics{}
    Saved to outputs/qc/<video_id>.json (unless --no-save).

CLI:
    python scripts/qc_video.py --input path/to/video.mp4
    python scripts/qc_video.py --input video.mp4 --min-duration 5 --brightness-threshold 30

Pipeline integration:
    if qc_video.qc_video(...).get("status") == "PASS":
        extract_frames(...)
        infer(...)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "rl_yolov8n_v1" / "weights" / "best.pt"
QC_DIR = ROOT / "outputs" / "qc"

# Defaults — tuned for short return-inspection clips, override via CLI flags.
DEFAULTS = {
    "sample_count": 10,
    "min_duration": 8.0,        # seconds
    "motion_min_diff": 2.0,     # mean abs pixel diff across consecutive samples (0-255)
    "brightness_threshold": 40.0,
    "blur_threshold": 100.0,    # Laplacian variance; below = blurry
    "blurry_frame_ratio": 0.6,  # if >this fraction of samples is blurry, fail blur rule
    "yolo_conf": 0.20,
    "min_detection_area": 0.02, # rel. to frame area; smaller mean detection -> occlusion
    "framing_min_area": 0.01,   # largest box smaller than this -> bad framing
}

CRITICAL_REASONS = {"too_short", "no_motion", "no_signal", "blurry"}


def derive_video_id(video_path: Path) -> str:
    stem = video_path.stem.split()[0] if video_path.stem else video_path.stem
    cleaned = re.sub(r"[^A-Za-z0-9\-]", "", stem)
    cleaned = cleaned.replace("--", "-").strip("-")
    return cleaned or "unknown"


def sample_frames(video_path: Path, n: int) -> tuple[list[np.ndarray], list[int], float, int, float]:
    """Return (frames, frame_indices, src_fps, total_frames, duration_sec)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps if src_fps > 0 else 0.0
    if total <= 0:
        cap.release()
        return [], [], src_fps, 0, 0.0

    n = max(1, min(n, total))
    indices = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)] if n > 1 else [total // 2]
    frames: list[np.ndarray] = []
    kept_idx: list[int] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            kept_idx.append(idx)
    cap.release()
    return frames, kept_idx, src_fps, total, duration


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def frame_to_frame_motion(frames_gray: list[np.ndarray]) -> float:
    """Mean of absolute differences between consecutive sampled frames."""
    if len(frames_gray) < 2:
        return 0.0
    diffs = []
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        diffs.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def run_yolo(frames: list[np.ndarray], weights: Path, conf: float) -> list[list[tuple[int, float, tuple[float, float, float, float]]]]:
    """Return per-frame list of (class_id, conf, (x1,y1,x2,y2)) detections."""
    if not weights.exists():
        print(f"WARN: weights not found at {weights} — signal rule will be skipped.", file=sys.stderr)
        return [[] for _ in frames]
    # Lazy import to keep --help fast.
    from ultralytics import YOLO
    model = YOLO(str(weights))
    results = model.predict(frames, conf=conf, verbose=False)
    out: list[list[tuple[int, float, tuple[float, float, float, float]]]] = []
    for r in results:
        per_frame = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                per_frame.append((int(cls[i]), float(cf[i]), (x1, y1, x2, y2)))
        out.append(per_frame)
    return out


def qc_video(
    video_path: Path,
    weights: Path = DEFAULT_WEIGHTS,
    sample_count: int = DEFAULTS["sample_count"],
    min_duration: float = DEFAULTS["min_duration"],
    motion_min_diff: float = DEFAULTS["motion_min_diff"],
    brightness_threshold: float = DEFAULTS["brightness_threshold"],
    blur_threshold: float = DEFAULTS["blur_threshold"],
    blurry_frame_ratio: float = DEFAULTS["blurry_frame_ratio"],
    yolo_conf: float = DEFAULTS["yolo_conf"],
    min_detection_area: float = DEFAULTS["min_detection_area"],
    framing_min_area: float = DEFAULTS["framing_min_area"],
    save: bool = True,
) -> dict[str, Any]:
    video_path = Path(video_path)
    video_id = derive_video_id(video_path)
    reasons: list[str] = []

    frames, _, src_fps, total_frames, duration = sample_frames(video_path, sample_count)
    if total_frames == 0 or not frames:
        result = {
            "video": str(video_path),
            "video_id": video_id,
            "status": "FAIL",
            "reasons": ["unreadable"],
            "metrics": {"duration": 0.0, "frames_sampled": 0},
        }
        if save:
            _save(result)
        return result

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = frames[0].shape[:2]
    frame_area = float(H * W)

    # (A) duration
    if duration < min_duration:
        reasons.append("too_short")

    # (B) motion
    motion_score = frame_to_frame_motion(grays)
    if motion_score < motion_min_diff:
        reasons.append("no_motion")

    # (D) blur
    blur_scores = [laplacian_variance(g) for g in grays]
    blurry_count = sum(1 for s in blur_scores if s < blur_threshold)
    blurry_ratio = blurry_count / len(blur_scores)
    if blurry_ratio > blurry_frame_ratio:
        reasons.append("blurry")

    # (E) brightness
    brightness_scores = [mean_brightness(g) for g in grays]
    avg_brightness = float(np.mean(brightness_scores))
    if avg_brightness < brightness_threshold:
        reasons.append("too_dark")

    # (C) signal — YOLO
    yolo_dets = run_yolo(frames, weights, yolo_conf)
    total_detections = sum(len(d) for d in yolo_dets)
    if total_detections == 0:
        reasons.append("no_signal")

    # (F)(G) occlusion + framing — only meaningful if we have detections
    largest_box_area_rel = 0.0
    mean_box_area_rel = 0.0
    if total_detections > 0:
        all_areas_rel = []
        for per_frame in yolo_dets:
            for _, _, (x1, y1, x2, y2) in per_frame:
                area = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
                all_areas_rel.append(area)
        if all_areas_rel:
            largest_box_area_rel = max(all_areas_rel)
            mean_box_area_rel = float(np.mean(all_areas_rel))
        if mean_box_area_rel > 0 and mean_box_area_rel < min_detection_area:
            reasons.append("occluded")
        if largest_box_area_rel > 0 and largest_box_area_rel < framing_min_area:
            reasons.append("bad_framing")

    status = "FAIL" if any(r in CRITICAL_REASONS for r in reasons) else "PASS"

    result = {
        "video": str(video_path),
        "video_id": video_id,
        "status": status,
        "reasons": reasons,
        "metrics": {
            "duration": round(duration, 2),
            "fps": round(src_fps, 2),
            "total_frames": total_frames,
            "frames_sampled": len(frames),
            "brightness": round(avg_brightness, 2),
            "motion_score": round(motion_score, 3),
            "blur_score_mean": round(float(np.mean(blur_scores)), 1),
            "blurry_frame_ratio": round(blurry_ratio, 2),
            "detections": int(total_detections),
            "largest_box_area_rel": round(largest_box_area_rel, 4),
            "mean_box_area_rel": round(mean_box_area_rel, 4),
        },
    }
    if save:
        _save(result)
    return result


def _save(result: dict) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QC_DIR / f"{result['video_id']}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _print_human(result: dict) -> None:
    status = result["status"]
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print("=" * 64)
    print(f"  Video    : {result['video']}")
    print(f"  video_id : {result['video_id']}")
    print(f"  Status   : {color}{status}{reset}")
    if result["reasons"]:
        critical = [r for r in result["reasons"] if r in CRITICAL_REASONS]
        warnings = [r for r in result["reasons"] if r not in CRITICAL_REASONS]
        if critical:
            print(f"  CRITICAL : {', '.join(critical)}")
        if warnings:
            print(f"  Warnings : {', '.join(warnings)}")
    print("  Metrics  :")
    for k, v in result["metrics"].items():
        print(f"    {k:<22} {v}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Path to source video")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help=f"YOLO weights (default: {DEFAULT_WEIGHTS})")
    ap.add_argument("--sample-count", type=int, default=DEFAULTS["sample_count"])
    ap.add_argument("--min-duration", type=float, default=DEFAULTS["min_duration"])
    ap.add_argument("--brightness-threshold", type=float, default=DEFAULTS["brightness_threshold"])
    ap.add_argument("--blur-threshold", type=float, default=DEFAULTS["blur_threshold"])
    ap.add_argument("--motion-min-diff", type=float, default=DEFAULTS["motion_min_diff"])
    ap.add_argument("--yolo-conf", type=float, default=DEFAULTS["yolo_conf"])
    ap.add_argument("--no-save", action="store_true", help="Skip writing JSON to outputs/qc/")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Video not found: {args.input}")

    result = qc_video(
        video_path=args.input,
        weights=args.weights,
        sample_count=args.sample_count,
        min_duration=args.min_duration,
        motion_min_diff=args.motion_min_diff,
        brightness_threshold=args.brightness_threshold,
        blur_threshold=args.blur_threshold,
        yolo_conf=args.yolo_conf,
        save=not args.no_save,
    )
    _print_human(result)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
