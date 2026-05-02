"""Video-stratified 80/20 train/val split (NO leakage).

Groups images by video_id (the prefix before "_f") and ensures all frames from
the same video go to either train OR val — never both.

Image stem convention (REQUIRED): <video_id>_f<NNNN>.jpg

Usage:
    # Damage (default)
    python scripts/split.py
    # Shipping label
    python scripts/split.py --data dataset/shipping_label/data.yaml
    # Custom source folder
    python scripts/split.py --data <yaml> --source dataset/shipping_label/_to_label
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = ROOT / "configs" / "data.yaml"
SEED = 42
VAL_RATIO = 0.2

random.seed(SEED)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML,
                    help="Path to YOLO data.yaml. Drives both source and target folders.")
    ap.add_argument("--source", type=Path, default=None,
                    help="Override source-image folder. "
                         "If omitted, auto-detects <data_root>/_raw or <data_root>/_to_label.")
    return ap.parse_args()


def _resolve_data_root(data_yaml: Path) -> Path:
    """Read data.yaml's `path` field; resolve project-root-relative or yaml-relative."""
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise SystemExit(f"Could not parse {data_yaml}: {e}")
    path_field = cfg.get("path", ".")
    for cand in ((ROOT / path_field), (data_yaml.parent / path_field)):
        r = cand.resolve()
        if r.exists():
            return r
    raise SystemExit(
        f"data.yaml `path: {path_field}` does not resolve to an existing folder.\n"
        f"Tried: {(ROOT / path_field).resolve()} and {(data_yaml.parent / path_field).resolve()}"
    )


def _resolve_source_dir(data_root: Path, override: Path | None) -> Path:
    """Find the folder containing labeled source frames.

    For versioned datasets, the data_root IS a version folder containing
    images at root level — handled by the final fallback below.
    """
    if override is not None:
        if not override.exists():
            raise SystemExit(f"--source folder not found: {override.resolve()}")
        return override.resolve()
    # Pre-versioning layout: a `_raw` or `_to_label` subfolder of the data root.
    for name in ("_raw", "_to_label"):
        cand = data_root / name
        if cand.exists() and (any(cand.glob("*.jpg")) or any(cand.glob("*.png"))):
            return cand
    # Versioned layout: data_root is itself the folder of frames (e.g. a
    # `v<date>_<note>/` folder, or v_legacy = the base folder).
    if any(data_root.glob("*.jpg")) or any(data_root.glob("*.png")):
        return data_root
    raise SystemExit(
        f"Could not auto-detect source frames. Looked in:\n"
        f"  {data_root / '_raw'}\n"
        f"  {data_root / '_to_label'}\n"
        f"  {data_root}/*.jpg | *.png  (versioned layout)\n"
        f"Pass --source <folder> explicitly."
    )


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    data_root = _resolve_data_root(data_yaml)
    source = _resolve_source_dir(data_root, args.source)

    print(f"data.yaml      : {data_yaml}")
    print(f"data root      : {data_root}")
    print(f"source frames  : {source}")
    print()

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        target = data_root / sub
        target.mkdir(parents=True, exist_ok=True)
        for f in target.iterdir():
            if f.is_file():
                f.unlink()

    # Honor the per-version exclude list if it exists. Excluded frames
    # never make it into the train/val split.
    excluded: set[str] = set()
    excl_file = source / "_excluded.json"
    if excl_file.exists():
        try:
            import json as _json
            excluded = set(_json.loads(excl_file.read_text(encoding="utf-8")))
        except Exception:
            excluded = set()

    images = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
    if excluded:
        before = len(images)
        images = [p for p in images if p.name not in excluded]
        print(f"excluded      : {before - len(images)} frame(s) per _excluded.json")
    if not images:
        raise SystemExit(f"No images found in {source}")

    videos: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        if "_f" not in img.stem:
            print(f"SKIP (bad name, missing '_f'): {img.name}", file=sys.stderr)
            continue
        video_id = img.stem.rsplit("_f", 1)[0]
        videos[video_id].append(img)

    if not videos:
        raise SystemExit("No images with valid <video_id>_f<NNNN> naming.")

    video_ids = sorted(videos.keys())
    if len(video_ids) < 2:
        print("=" * 64, file=sys.stderr)
        print(f"ERROR: only {len(video_ids)} video(s) found: {video_ids}", file=sys.stderr)
        print("Need at least 2 videos for a leak-free train/val split.", file=sys.stderr)
        print("Add more videos via scripts/extract_frames.py and rerun.", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
        raise SystemExit(2)

    random.shuffle(video_ids)
    val_cut = max(1, int(round(len(video_ids) * VAL_RATIO)))
    val_videos = set(video_ids[:val_cut])
    train_videos = [v for v in video_ids if v not in val_videos]

    counts = {"train_imgs": 0, "val_imgs": 0, "labeled": 0, "unlabeled": 0}
    for video_id, imgs in videos.items():
        split = "val" if video_id in val_videos else "train"
        for img in imgs:
            shutil.copy2(img, data_root / "images" / split / img.name)
            label = source / f"{img.stem}.txt"
            if label.exists() and label.stat().st_size > 0:
                shutil.copy2(label, data_root / "labels" / split / label.name)
                counts["labeled"] += 1
            else:
                counts["unlabeled"] += 1
            counts[f"{split}_imgs"] += 1

    print("=" * 64)
    print(f"videos total  : {len(video_ids)}")
    print(f"  train       : {len(train_videos)}  {sorted(train_videos)}")
    print(f"  val         : {len(val_videos)}  {sorted(val_videos)}")
    print(f"images train  : {counts['train_imgs']}")
    print(f"images val    : {counts['val_imgs']}")
    print(f"with labels   : {counts['labeled']}")
    print(f"no label file : {counts['unlabeled']}  (treated as negatives)")
    print("=" * 64)


if __name__ == "__main__":
    main()
