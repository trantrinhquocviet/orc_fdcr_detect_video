"""Push .txt label files from dataset/_to_label/ back into dataset/_raw/.

Run this after labeling in the staged folder. Existing labels in _raw are
overwritten (newer label wins).
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "_raw"
TO_LABEL = ROOT / "dataset" / "_to_label"


def main() -> None:
    if not TO_LABEL.exists():
        raise SystemExit(f"Staged folder not found: {TO_LABEL}")

    txt_files = sorted(TO_LABEL.glob("*.txt"))
    if not txt_files:
        raise SystemExit("No .txt label files in _to_label — did you save in LabelImg?")

    skipped_classes_txt = 0
    copied = 0
    for src in txt_files:
        # LabelImg writes a 'classes.txt' that we don't want to merge.
        if src.name == "classes.txt":
            skipped_classes_txt += 1
            continue
        dst = RAW / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f"Merged {copied} label file(s) into {RAW.relative_to(ROOT)}")
    if skipped_classes_txt:
        print(f"Skipped {skipped_classes_txt} 'classes.txt' file(s).")


if __name__ == "__main__":
    main()
