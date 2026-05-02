"""Build a focused labeling folder from triage CSV.

Reads outputs/review/review__raw.csv (one CSV row per yes/no decision, last
vote wins per image), picks up to MAX_PER_VIDEO yes-frames per video_id with
even spacing, copies them into dataset/_to_label/.

After labeling there, run scripts/merge_labels.py to push .txt files back
into dataset/_raw.
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "_raw"
TO_LABEL = ROOT / "dataset" / "_to_label"
CSV_PATH = ROOT / "outputs" / "review" / "review__raw.csv"
MAX_PER_VIDEO = 10


def even_sample(items: list, k: int) -> list:
    """Pick k items spread evenly across a sorted list."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Triage CSV not found: {CSV_PATH}")

    last: dict[str, dict] = {}
    for row in csv.DictReader(CSV_PATH.open(encoding="utf-8")):
        last[row["image_name"]] = row

    yes_per_video: dict[str, list[str]] = defaultdict(list)
    for img_name, row in last.items():
        if row["is_correct"].strip().lower() == "yes":
            yes_per_video[row["video_id"]].append(img_name)

    if TO_LABEL.exists():
        for f in TO_LABEL.iterdir():
            if f.is_file():
                f.unlink()
    else:
        TO_LABEL.mkdir(parents=True)

    print(f"Source CSV    : {CSV_PATH.relative_to(ROOT)}")
    print(f"Target folder : {TO_LABEL.relative_to(ROOT)}")
    print(f"Cap per video : {MAX_PER_VIDEO}")
    print("=" * 64)

    total = 0
    for video_id in sorted(yes_per_video):
        candidates = sorted(yes_per_video[video_id])
        picked = even_sample(candidates, MAX_PER_VIDEO)
        for name in picked:
            src = RAW / name
            if not src.exists():
                print(f"  MISSING: {name}")
                continue
            shutil.copy2(src, TO_LABEL / name)
            # If a label already exists in _raw, copy it too so LabelImg shows
            # existing boxes and the user can refine instead of redoing.
            existing_label = RAW / f"{src.stem}.txt"
            if existing_label.exists():
                shutil.copy2(existing_label, TO_LABEL / f"{src.stem}.txt")
        print(f"  {video_id:<22} picked {len(picked):>2} of {len(candidates)} yes-frames")
        total += len(picked)

    print("=" * 64)
    print(f"Total staged for labeling: {total} frames")
    print()
    print("Next:")
    print("  Open LabelImg on the staged folder:")
    print(f'    .venv/Scripts/python.exe .venv/Lib/site-packages/labelImg/labelImg.py '
          f'dataset/_to_label configs/predefined_classes.txt dataset/_to_label')


if __name__ == "__main__":
    main()
