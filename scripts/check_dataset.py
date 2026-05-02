"""Quick sanity check on a YOLO-format dataset folder.

Verifies that:
  <dataset>/images/{train,val} contain images
  <dataset>/labels/{train,val} contain matching .txt files (same stem)
  no labels are empty (size 0)

Run before train.py to catch path / pairing issues early.

Usage:
    python scripts/check_dataset.py --data dataset/shipping_label
    python scripts/check_dataset.py --data configs/data.yaml   # also accepts a YAML
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_dataset_dir(arg: Path) -> Path:
    """Accept either a dataset folder OR a YOLO data.yaml. Return the folder."""
    p = arg.resolve()
    if p.is_file() and p.suffix in (".yml", ".yaml"):
        try:
            import yaml
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"  [error] could not parse YAML: {e}")
            return p.parent
        path_field = cfg.get("path")
        if path_field:
            cand = (ROOT / path_field).resolve()
            if cand.exists():
                return cand
            cand2 = (p.parent / path_field).resolve()
            if cand2.exists():
                return cand2
            return cand  # let the downstream check report it
        return p.parent
    return p


def check(dataset_dir: Path) -> int:
    issues: list[str] = []
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            print(f"  [{split}] MISSING images dir: {img_dir.relative_to(ROOT) if img_dir.is_relative_to(ROOT) else img_dir}")
            issues.append(f"missing_img_dir_{split}")
            continue

        imgs = sorted(p for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
        lbls = (sorted(p for p in lbl_dir.iterdir()
                       if p.is_file() and p.suffix == ".txt")
                if lbl_dir.exists() else [])

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        unmatched_imgs = sorted(img_stems - lbl_stems)
        unmatched_lbls = sorted(lbl_stems - img_stems)
        empty_lbls = sorted(p.name for p in lbls if p.stat().st_size == 0)

        counts[split] = len(imgs)
        # Real errors: orphan labels (no matching image) or empty .txt files.
        # Images without a label are NOT errors — YOLO treats them as
        # negative samples (no objects). Reported as info only.
        clean = not (unmatched_lbls or empty_lbls)
        status = "OK" if clean else "MISMATCH"
        print(f"  [{split}] {len(imgs)} images / {len(lbls)} labels — {status}")

        if unmatched_imgs:
            print(f"      info: {len(unmatched_imgs)} image(s) have no label "
                  f"(treated as negatives by YOLO)")
        if unmatched_lbls:
            print(f"      ERROR: {len(unmatched_lbls)} label(s) have no matching image "
                  f"(e.g. {unmatched_lbls[:3]})")
            issues.append(f"unmatched_lbls_{split}")
        if empty_lbls:
            print(f"      ERROR: {len(empty_lbls)} empty label file(s) "
                  f"(e.g. {empty_lbls[:3]})")
            issues.append(f"empty_lbls_{split}")

    if issues:
        print(f"\n  Result: {len(issues)} issue(s) — fix before training.")
        return 1
    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0:
        print("\n  Result: pairs OK but at least one split is EMPTY — "
              "run scripts/split.py to populate images/{train,val}.")
        return 3
    print("\n  Result: dataset OK — ready to train.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to dataset folder OR a YOLO data.yaml file.")
    args = ap.parse_args()

    folder = _resolve_dataset_dir(args.data)
    print(f"Dataset folder: {folder}")
    print(f"Folder exists : {folder.exists()}")
    if not folder.exists():
        return 2
    return check(folder)


if __name__ == "__main__":
    raise SystemExit(main())
