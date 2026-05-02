"""Extract frames from a video at a target FPS using OpenCV.

Output filename convention (ENFORCED):
    <video_id>_f<NNNN>.jpg

Why: enables case-aware train/val splitting and traceability back to source.

Usage:
    # auto-derives video_id from filename stem
    python scripts/extract_frames.py --video path/to/TTKMFB581.mp4 --fps 1

    # explicit video_id (recommended for messy filenames)
    python scripts/extract_frames.py --video "path with spaces.mp4" --video-id RT00231 --fps 1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "_raw"

# video_id rules: alphanumeric + dash + underscore only, no "_f" suffix collisions
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]*$")


def derive_video_id(video_path: Path) -> str:
    """Auto-derive a contract-safe video_id from the filename stem.

    Steps:
      1. Strip Vietnamese / European accents via NFKD (e.g. "Cần" → "Can").
         "đ"/"Đ" don't decompose this way — handled explicitly first.
      2. ASCII-fold; drop anything that survived (rare).
      3. Replace any run of non-[A-Za-z0-9] with a single dash. This catches
         spaces, punctuation, and any leftover non-ASCII marks.
      4. Trim leading/trailing dashes.

    Result is alphanumeric + dashes only — exactly the filename contract.
    Underscores are NOT used because `_f` is the frame-index delimiter parsers
    rely on; an underscore in the video_id breaks split.py / parse_video_id.

    Examples:
      "VD KH TTKMFB-583030068944012504"   -> "VD-KH-TTKMFB-583030068944012504"
      "TTVN1060812452"                    -> "TTVN1060812452"
      "Cần Thơ video"                     -> "Can-Tho-video"
      "đêm test 1"                        -> "dem-test-1"
    """
    import unicodedata
    s = video_path.stem
    s = s.replace("đ", "d").replace("Đ", "D").replace("ð", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    if not s:
        raise SystemExit(
            f"Could not derive video_id from '{video_path.name}'. "
            "Pass --video-id explicitly."
        )
    return s


def validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_PATTERN.match(video_id):
        raise SystemExit(
            f"Invalid video_id '{video_id}'. Use alphanumeric + dashes only, no underscores."
        )
    if "_f" in video_id:
        raise SystemExit(f"video_id must not contain '_f' (collides with frame suffix).")


def extract(
    video_path: Path,
    video_id: str,
    target_fps: float,
    width: int | None,
    output_dir: Path | None = None,
) -> int:
    out_dir = Path(output_dir) if output_dir is not None else RAW_DIR
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / target_fps)))

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            saved += 1
            if width and frame.shape[1] != width:
                h = int(frame.shape[0] * (width / frame.shape[1]))
                frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{video_id}_f{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        idx += 1

    cap.release()
    print(f"video         : {video_path.name}")
    print(f"video_id      : {video_id}")
    print(f"source fps    : {src_fps:.2f}")
    print(f"total frames  : {total}")
    print(f"sampling step : every {step} frames (~{target_fps} fps)")
    print(f"saved frames  : {saved} -> {out_dir}")
    print(f"naming        : {video_id}_f0001.jpg ... {video_id}_f{saved:04d}.jpg")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path, help="Path to source video")
    ap.add_argument("--video-id", "--video_id", dest="video_id", default=None,
                    help="Video identifier (auto-derived from filename if omitted)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=None,
                    help="Resize frames to this width (height auto). Skip to keep source res.")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    video_id = args.video_id or derive_video_id(args.video)
    validate_video_id(video_id)
    extract(args.video, video_id, args.fps, args.width)


if __name__ == "__main__":
    main()
