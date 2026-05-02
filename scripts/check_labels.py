"""Pre-training quality check for YOLO labels.

Scans dataset/_raw (or any folder) and reports:
  - images without label files (will be treated as negatives)
  - empty label files
  - invalid class IDs (must be 0 or 1 for MVP)
  - malformed lines (wrong number of fields, out-of-range coords)
  - boxes that look too small (likely accidental clicks)
  - boxes that cover most of the image (likely too loose)

Usage:
    python scripts/check_labels.py                       # checks dataset/_raw
    python scripts/check_labels.py --dir dataset/_raw    # explicit
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_CLASSES = {0, 1}  # damaged_item, empty_box

MIN_BOX_AREA = 0.001   # <0.1% of image -> probably misclick
MAX_BOX_AREA = 0.95    # >95% of image -> probably too loose


def check(folder: Path) -> int:
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not images:
        print(f"No images in {folder}")
        return 1

    n_total = len(images)
    n_labeled = 0
    n_unlabeled = 0
    issues: list[str] = []
    class_counts = {0: 0, 1: 0}

    for img in images:
        label = img.with_suffix(".txt")
        if not label.exists():
            n_unlabeled += 1
            continue

        text = label.read_text(encoding="utf-8").strip()
        if not text:
            issues.append(f"EMPTY     {label.name}  (no boxes — delete the .txt or label the image)")
            continue

        n_labeled += 1
        for ln, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"MALFORMED {label.name}:{ln}  wrong field count ({len(parts)} != 5)")
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = (float(p) for p in parts[1:])
            except ValueError:
                issues.append(f"MALFORMED {label.name}:{ln}  non-numeric values")
                continue

            if cls not in VALID_CLASSES:
                issues.append(f"BAD CLASS {label.name}:{ln}  class={cls} (must be 0 or 1)")
                continue
            class_counts[cls] += 1

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                issues.append(f"OUT OF RANGE {label.name}:{ln}  x={x} y={y} w={w} h={h}")
                continue

            area = w * h
            if area < MIN_BOX_AREA:
                issues.append(f"TINY BOX  {label.name}:{ln}  area={area:.4f} (likely misclick)")
            elif area > MAX_BOX_AREA:
                issues.append(f"HUGE BOX  {label.name}:{ln}  area={area:.4f} (covers whole frame)")

    print("=" * 60)
    print(f"Folder            : {folder}")
    print(f"Images total      : {n_total}")
    print(f"With labels       : {n_labeled}")
    print(f"Without labels    : {n_unlabeled}  (treated as negatives during training)")
    print(f"Class 0 (damaged) : {class_counts[0]} boxes")
    print(f"Class 1 (empty)   : {class_counts[1]} boxes")
    print(f"Issues found      : {len(issues)}")
    print("=" * 60)
    for line in issues:
        print(line)

    if class_counts[0] == 0 and class_counts[1] == 0:
        print("\nNo valid labels yet — go label some images.")
        return 1
    if not issues:
        print("\nAll good. Ready to split + train.")
    else:
        print(f"\nFix the {len(issues)} issue(s) above, then re-run this check.")
    return 0 if not issues else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "dataset" / "_raw")
    args = ap.parse_args()
    raise SystemExit(check(args.dir))


if __name__ == "__main__":
    main()
