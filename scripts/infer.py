"""Local inference pipeline for reverse-logistics image/video analysis.

Pipeline:
    file -> (if video) ffmpeg frame extraction -> YOLOv8 batch detection
         -> rule engine -> JSON output ready for an Electron frontend.

Usage:
    python scripts/infer.py --input path/to/file.mp4 --weights models/rl_yolov8n/weights/best.pt
    python scripts/infer.py --input path/to/folder/  # batch over a folder
    python scripts/infer.py --input file.jpg --weights yolov8n.pt --conf 0.35

Output JSON is written to outputs/json/<stem>.json and also printed to stdout
so an Electron host process can capture it directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

from rules import aggregate, classify_frame

ROOT = Path(__file__).resolve().parents[1]
OUT_FRAMES = ROOT / "outputs" / "frames"
OUT_JSON = ROOT / "outputs" / "json"
OUT_ANNOT = ROOT / "outputs" / "annotated"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# ---------- frame extraction ----------

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames(video: Path, fps: float, out_dir: Path) -> list[Path]:
    """Extract frames from a video at `fps` frames per second.

    Uses ffmpeg if available (faster, reliable on Windows); falls back to
    OpenCV so the pipeline still runs in lean environments.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%05d.jpg"

    if have_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(pattern),
        ]
        subprocess.run(cmd, check=True)
    else:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(src_fps / fps)), 1)
        idx, saved = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                saved += 1
                cv2.imwrite(str(out_dir / f"frame_{saved:05d}.jpg"), frame)
            idx += 1
        cap.release()

    return sorted(out_dir.glob("frame_*.jpg"))


# ---------- detection ----------

def run_detection(
    model: YOLO,
    frames: list[Path],
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
) -> list[dict]:
    """Batch YOLO inference. Returns list of per-frame dicts."""
    results_out: list[dict] = []
    names = model.names  # {id: name}

    for start in range(0, len(frames), batch):
        chunk = frames[start : start + batch]
        # stream=False: we want all results back together for this chunk.
        results = model.predict(
            source=[str(p) for p in chunk],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for path, r in zip(chunk, results):
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                results_out.append({
                    "frame": path.name,
                    "detections": [],
                    "max_conf": 0.0,
                })
                continue

            cls_ids = boxes.cls.int().tolist()
            confs = boxes.conf.float().tolist()
            xyxy = boxes.xyxy.float().tolist()

            dets = [
                {
                    "class": names[c],
                    "confidence": round(float(p), 4),
                    "bbox": [round(v, 2) for v in box],
                }
                for c, p, box in zip(cls_ids, confs, xyxy)
            ]
            results_out.append({
                "frame": path.name,
                "detections": dets,
                "max_conf": round(max(confs), 4) if confs else 0.0,
            })
    return results_out


# ---------- per-file orchestration ----------

def analyze_file(
    file: Path,
    model: YOLO,
    fps: float,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    save_annotated: bool,
) -> dict:
    t0 = time.time()
    ext = file.suffix.lower()
    is_video = ext in VIDEO_EXTS
    is_image = ext in IMG_EXTS
    if not (is_video or is_image):
        raise ValueError(f"Unsupported file type: {file}")

    if is_video:
        frame_dir = OUT_FRAMES / file.stem
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frames = extract_frames(file, fps=fps, out_dir=frame_dir)
    else:
        frames = [file]

    if not frames:
        return {
            "file": file.name,
            "type": "video" if is_video else "image",
            "verdict": "ok",
            "issues_detected": [],
            "frames_flagged": [],
            "confidence": 0.0,
            "frame_count": 0,
            "elapsed_sec": round(time.time() - t0, 3),
            "error": "no_frames_extracted",
        }

    per_frame = run_detection(model, frames, conf, iou, imgsz, batch, device)

    # Map raw class names -> business issues per frame.
    per_frame_issues: list[list[str]] = []
    flagged: list[str] = []
    confs_for_flagged: list[float] = []
    for entry in per_frame:
        class_names = [d["class"] for d in entry["detections"]]
        issues = classify_frame(class_names)
        entry["issues"] = issues
        per_frame_issues.append(issues)
        if issues != ["ok"]:
            flagged.append(entry["frame"])
            confs_for_flagged.append(entry["max_conf"])

    summary = aggregate(per_frame_issues)
    overall_conf = (
        round(sum(confs_for_flagged) / len(confs_for_flagged), 4)
        if confs_for_flagged else 0.0
    )

    if save_annotated and is_image:
        annot = model.predict(
            source=str(file), conf=conf, iou=iou, imgsz=imgsz,
            device=device, verbose=False, save=True,
            project=str(OUT_ANNOT), name=file.stem, exist_ok=True,
        )
        del annot

    return {
        "file": file.name,
        "type": "video" if is_video else "image",
        "verdict": summary["verdict"],
        "issues_detected": summary["issues_detected"],
        "issue_counts": summary["issue_counts"],
        "frames_flagged": flagged,
        "confidence": overall_conf,
        "frame_count": len(frames),
        "frames_sampled_fps": fps if is_video else None,
        "per_frame": per_frame,
        "elapsed_sec": round(time.time() - t0, 3),
    }


# ---------- CLI ----------

def iter_inputs(target: Path) -> Iterable[Path]:
    if target.is_file():
        yield target
        return
    if target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.suffix.lower() in IMG_EXTS | VIDEO_EXTS:
                yield p
        return
    raise FileNotFoundError(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-logistics local inference.")
    parser.add_argument("--input", required=True, help="Image, video, or folder.")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Path to .pt weights (default: stock yolov8n.pt).")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for frame inference.")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Sampling rate for video frame extraction (frames/sec).")
    parser.add_argument("--device", default="cpu",
                        help="'cpu', '0' for first GPU, etc.")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Also save YOLO-annotated images under outputs/annotated/.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stdout JSON (file is still written).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.input).expanduser().resolve()

    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_ANNOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    all_results = []
    for f in iter_inputs(target):
        try:
            result = analyze_file(
                file=f,
                model=model,
                fps=args.fps,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                save_annotated=args.save_annotated,
            )
        except Exception as exc:  # noqa: BLE001 — surface failure in JSON, keep going.
            result = {
                "file": f.name,
                "type": "unknown",
                "verdict": "error",
                "error": str(exc),
            }

        out_path = OUT_JSON / f"{f.stem}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results.append(result)

    if not args.quiet:
        # Single JSON document for easy ingestion by Electron's child_process.
        json.dump(
            all_results if len(all_results) != 1 else all_results[0],
            sys.stdout, indent=2, ensure_ascii=False,
        )
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
