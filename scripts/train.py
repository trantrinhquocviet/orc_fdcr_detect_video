"""Train a YOLOv8 model on the reverse-logistics dataset.

Usage:
    # Damage (default)
    python scripts/train.py --epochs 100 --imgsz 640
    # Shipping label
    python scripts/train.py --data dataset/shipping_label/data.yaml --name rl_shipping_label_v1

Notes:
- Start from yolov8n.pt (nano) for fast local training and inference.
- Step up to yolov8s.pt / yolov8m.pt only if recall is too low after 100+ epochs.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = ROOT / "configs" / "data.yaml"
MODELS_DIR = ROOT / "models"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for reverse logistics.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML,
                        help="Path to YOLO data.yaml. Defaults to configs/data.yaml (Damage).")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU.")
    parser.add_argument("--name", default="rl_yolov8n", help="Run name under models/.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    return parser.parse_args()


def _resolve_dataset_root(data_yaml: Path) -> Path | None:
    """Read data.yaml's `path` field and resolve it. Returns None on failure."""
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    path_field = cfg.get("path")
    if not path_field:
        return data_yaml.parent
    # Try project-root-relative first, then yaml-relative.
    for candidate in ((ROOT / path_field), (data_yaml.parent / path_field)):
        cand = candidate.resolve()
        if cand.exists():
            return cand
    return (ROOT / path_field).resolve()


def _print_dataset_debug(data_yaml: Path) -> None:
    """Spec §4: print absolute path + image counts before training."""
    abs_yaml = data_yaml.resolve()
    print("=" * 60)
    print(f"[train] data.yaml         : {abs_yaml}")

    root = _resolve_dataset_root(data_yaml)
    print(f"[train] dataset root      : {root}")
    if root is None or not root.exists():
        print(f"[train] WARNING dataset root does not exist or unreadable")
        print("=" * 60)
        return

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        n_img = sum(1 for p in img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS) if img_dir.exists() else 0
        n_lbl = sum(1 for p in lbl_dir.iterdir()
                    if p.is_file() and p.suffix == ".txt") if lbl_dir.exists() else 0
        flag = "" if (n_img > 0 and n_lbl > 0) else "  ⚠"
        print(f"[train] {split:<5}                : {n_img} images / {n_lbl} labels{flag}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    _print_dataset_debug(args.data)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(MODELS_DIR),
        name=args.name,
        patience=args.patience,
        # Augmentation defaults are sensible; tweak only if dataset is tiny.
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Save best + last; export to ONNX for portability into Electron later.
        save=True,
    )

    # Export the best weights to ONNX (cross-platform, easy to ship in Electron).
    best = MODELS_DIR / args.name / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).export(format="onnx", opset=12, simplify=True)
        print(f"[train] Exported ONNX next to: {best}")


if __name__ == "__main__":
    main()
