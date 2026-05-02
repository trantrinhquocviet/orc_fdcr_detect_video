"""Tracking code visibility check — detect shipping label and OCR tracking text.

Pipeline:
    (A) sample N frames evenly
    (B) YOLO -> top "shipping_label" box per frame (conf >= --conf)
    (C) crop with padding (skip frames with no detection)
    (D) OCR crop (EasyOCR by default, fallback Tesseract via --ocr tesseract)
    (E) clean text -> validate (len>=8 and digit_count>=6)

Decision (MVP):
    By default (--strict): label_detected >= min AND valid_ocr >= min -> PASS
    With --ocr-only: valid_ocr >= --min-valid-ocr -> PASS  (label detection ignored
        for the decision; useful while shipping_label weights don't exist yet).

Pipeline integration (caller side):
    if qc_video(...).status == "PASS":
        if check_tracking(...).status == "PASS":
            extract_frames(...) / infer(...)
        else: reject(reason="tracking_unreadable")

CLI:
    python scripts/check_tracking.py --input video.mp4
    python scripts/check_tracking.py --input v.mp4 --label-class shipping_label
    python scripts/check_tracking.py --input v.mp4 --save-crops --ocr tesseract

NOTE: The current MVP weights (rl_yolov8n_v1/v2) only contain damaged_item and
empty_box. If --label-class is not present in the model's class list, the script
runs OCR on the whole frame (warning printed) — useful as a smoke test until a
shipping_label-aware model is trained.
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
DEFAULT_WEIGHTS = ROOT / "models" / "rl_yolov8n_v2" / "weights" / "best.pt"
TRACK_DIR = ROOT / "outputs" / "tracking"

DEFAULTS = {
    "sample_count": 12,
    "conf": 0.30,
    "label_class": "shipping_label",
    "crop_pad": 0.08,            # fraction of bbox size, each side
    "min_frames_detected": 2,
    "min_valid_ocr": 1,
    "ocr": "easyocr",
    "min_text_len": 8,
    "min_digit_count": 6,
    "ocr_only": True,  # MVP: prioritize OCR signal while shipping_label model is absent
}

DEBUG_DIR = ROOT / "outputs" / "tracking" / "debug"

_KEEP = re.compile(r"[^A-Z0-9]")


def derive_video_id(video_path: Path) -> str:
    stem = video_path.stem.split()[0] if video_path.stem else video_path.stem
    cleaned = re.sub(r"[^A-Za-z0-9\-]", "", stem)
    cleaned = cleaned.replace("--", "-").strip("-")
    return cleaned or "unknown"


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    n = max(1, min(n, total))
    idxs = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)] if n > 1 else [total // 2]
    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def resolve_class_id(model, class_name: str) -> int | None:
    """Return class_id for given class_name in model.names, else None."""
    names = getattr(model, "names", {}) or {}
    for cid, cname in (names.items() if isinstance(names, dict) else enumerate(names)):
        if str(cname).lower() == class_name.lower():
            return int(cid)
    return None


def detect_label_boxes(model, frames: list[np.ndarray], class_id: int | None, conf: float):
    """Return list aligned to frames: best (x1,y1,x2,y2,conf) or None per frame."""
    results = model.predict(frames, conf=conf, verbose=False)
    out: list[tuple[float, float, float, float, float] | None] = []
    for r in results:
        best = None
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                if class_id is not None and int(cls[i]) != class_id:
                    continue
                c = float(cf[i])
                if best is None or c > best[4]:
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    best = (float(x1), float(y1), float(x2), float(y2), c)
        out.append(best)
    return out


def crop_with_padding(frame: np.ndarray, box, pad: float) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2, _ = box
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad, bh * pad
    xa = max(0, int(round(x1 - px)))
    ya = max(0, int(round(y1 - py)))
    xb = min(W, int(round(x2 + px)))
    yb = min(H, int(round(y2 + py)))
    return frame[ya:yb, xa:xb].copy()


def clean_text(s: str) -> str:
    return _KEEP.sub("", s.upper())


def validate(cleaned: str, min_len: int, min_digits: int) -> bool:
    if len(cleaned) < min_len:
        return False
    digits = sum(ch.isdigit() for ch in cleaned)
    return digits >= min_digits


# ---------- OCR backends (lazy imports) ----------

class _EasyOCR:
    def __init__(self) -> None:
        try:
            import easyocr  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "easyocr not installed. Run: .venv\\Scripts\\pip install easyocr\n"
                "(or use --ocr tesseract)"
            ) from e
        # English only — tracking codes are A-Z/0-9.
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, img: np.ndarray) -> list[str]:
        # detail=0 -> only text strings.
        return [str(t) for t in self.reader.readtext(img, detail=0, paragraph=False)]


class _Tesseract:
    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "pytesseract not installed. Run: .venv\\Scripts\\pip install pytesseract\n"
                "Tesseract binary required: https://github.com/UB-Mannheim/tesseract/wiki"
            ) from e
        self.pyt = pytesseract

    def read(self, img: np.ndarray) -> list[str]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        cfg = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = self.pyt.image_to_string(gray, config=cfg)
        return [line for line in text.splitlines() if line.strip()]


def make_ocr(name: str):
    name = name.lower()
    if name == "easyocr":
        return _EasyOCR()
    if name == "tesseract":
        return _Tesseract()
    raise SystemExit(f"Unknown --ocr backend: {name}")


# ---------- main check ----------

_DEFAULT_TRACKING_PATTERN = re.compile(r"^[A-Z]{2,8}[0-9]{6,18}$")


def check_tracking(
    video_path: Path,
    weights: Path = DEFAULT_WEIGHTS,
    sample_count: int = DEFAULTS["sample_count"],
    conf: float = DEFAULTS["conf"],
    label_class: str = DEFAULTS["label_class"],
    crop_pad: float = DEFAULTS["crop_pad"],
    min_frames_detected: int = DEFAULTS["min_frames_detected"],
    min_valid_ocr: int = DEFAULTS["min_valid_ocr"],
    ocr_backend: str = DEFAULTS["ocr"],
    min_text_len: int = DEFAULTS["min_text_len"],
    min_digit_count: int = DEFAULTS["min_digit_count"],
    save_crops: bool = False,
    ocr_only: bool = DEFAULTS["ocr_only"],
    save_debug: bool = True,
    strict_no_box_fallback: bool = False,
    tracking_pattern: str | None = None,
) -> dict[str, Any]:
    """
    strict_no_box_fallback: when True, OCR runs ONLY on detected box crops.
        Frames with no detection contribute zero OCR. If no frame has a box at
        all (or weights/class are missing), returns status="no_label_detected"
        without running OCR anywhere. Used by pipeline_runner.

    tracking_pattern: optional regex (post clean_text — uppercase alphanumeric)
        that valid OCR text must match. Defaults to the shipping-label-style
        pattern (^[A-Z]{2,8}[0-9]{6,18}$ — e.g. TTVN1064832858, TTKMFB...).
    """
    video_path = Path(video_path)
    video_id = derive_video_id(video_path)
    frames = sample_frames(video_path, sample_count)

    base = {
        "video": str(video_path),
        "video_id": video_id,
        "label_detected_frames": 0,
        "valid_ocr_frames": 0,
        "best_text": "",
        "status": "FAIL",
    }

    if not frames:
        base["reason"] = "unreadable"
        return base

    pattern: re.Pattern | None
    if tracking_pattern is None:
        pattern = _DEFAULT_TRACKING_PATTERN if strict_no_box_fallback else None
    else:
        pattern = re.compile(tracking_pattern)

    # Lazy-load YOLO. Strict mode short-circuits on missing weights/class.
    class_id: int | None = None
    if not weights.exists():
        if strict_no_box_fallback:
            base["status"] = "no_label_detected"
            base["reason"] = f"weights_missing: {weights}"
            return base
        print(f"WARN: weights not found at {weights} — running OCR on full frames.", file=sys.stderr)
        boxes_per_frame = [None] * len(frames)
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        class_id = resolve_class_id(model, label_class)
        if class_id is None:
            if strict_no_box_fallback:
                base["status"] = "no_label_detected"
                base["reason"] = (
                    f"class '{label_class}' not in model "
                    f"({list(getattr(model, 'names', {}).values())})"
                )
                return base
            print(
                f"WARN: class '{label_class}' not in model classes "
                f"({list(getattr(model, 'names', {}).values())}) — running OCR on full frames.",
                file=sys.stderr,
            )
            boxes_per_frame = [None] * len(frames)
        else:
            boxes_per_frame = detect_label_boxes(model, frames, class_id, conf)

    # In strict mode, instantiate the OCR backend lazily — only if we have at
    # least one detected box. Avoids the heavy easyocr cold-start when there's
    # nothing to OCR.
    ocr = None
    label_detected = 0
    valid_ocr = 0
    best = {"text": "", "score": -1.0, "frame_idx": -1, "crop": None}
    frames_log: list[dict[str, Any]] = []

    for i, (frame, box) in enumerate(zip(frames, boxes_per_frame)):
        if box is not None:
            label_detected += 1
            crop = crop_with_padding(frame, box, crop_pad)
            det_score = box[4]
            det_box = [round(v, 1) for v in box[:4]]
            source = "detected_crop"
        elif strict_no_box_fallback:
            # Spec: NO fallback OCR. Frame contributes a no-box log entry.
            frames_log.append({
                "frame": i, "source": "no_box", "det_conf": None,
                "det_box_xyxy": None, "raw_texts": [],
                "cleaned_candidates": [], "valid": False,
            })
            continue
        else:
            crop = frame
            det_score = 0.0
            det_box = None
            source = "full_frame"

        per_frame_entry: dict[str, Any] = {
            "frame": i,
            "source": source,
            "det_conf": round(det_score, 3) if det_box else None,
            "det_box_xyxy": det_box,
            "raw_texts": [],
            "cleaned_candidates": [],
            "valid": False,
        }

        if crop.size == 0:
            per_frame_entry["error"] = "empty_crop"
            frames_log.append(per_frame_entry)
            continue

        if ocr is None:
            ocr = make_ocr(ocr_backend)
        candidates = ocr.read(crop)
        per_frame_entry["raw_texts"] = list(candidates)

        frame_valid = False
        for raw in candidates:
            cleaned = clean_text(raw)
            digits = sum(ch.isdigit() for ch in cleaned)
            valid = validate(cleaned, min_text_len, min_digit_count)
            if valid and pattern is not None and not pattern.match(cleaned):
                valid = False  # pattern check overrides
            per_frame_entry["cleaned_candidates"].append({
                "cleaned": cleaned, "len": len(cleaned), "digits": digits,
                "valid": valid,
                "pattern_match": (
                    bool(pattern.match(cleaned)) if pattern is not None else None
                ),
            })
            if valid:
                score = len(cleaned) + det_score
                if score > best["score"]:
                    best = {"text": cleaned, "score": score, "frame_idx": i, "crop": crop}
                if not frame_valid:
                    valid_ocr += 1
                    frame_valid = True
        per_frame_entry["valid"] = frame_valid
        frames_log.append(per_frame_entry)

    if strict_no_box_fallback:
        if label_detected == 0:
            status = "no_label_detected"
            rule = "strict_no_box_fallback"
        elif valid_ocr >= min_valid_ocr:
            status = "PASS"
            rule = "strict_no_box_fallback"
        else:
            status = "FAIL"
            rule = "strict_no_box_fallback"
    elif ocr_only:
        status = "PASS" if valid_ocr >= min_valid_ocr else "FAIL"
        rule = "ocr_only"
    else:
        status = "PASS" if (label_detected >= min_frames_detected and valid_ocr >= min_valid_ocr) else "FAIL"
        rule = "strict"

    if save_crops and best["crop"] is not None:
        TRACK_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TRACK_DIR / f"{video_id}_best.jpg"
        cv2.imwrite(str(out_path), best["crop"])

    result = {
        "video": str(video_path),
        "video_id": video_id,
        "label_detected_frames": int(label_detected),
        "valid_ocr_frames": int(valid_ocr),
        "best_text": best["text"],
        "best_frame_idx": int(best["frame_idx"]),
        "decision_rule": rule,
        "status": status,
    }
    if save_debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / f"{video_id}.json").write_text(
            json.dumps({**result, "frames": frames_log}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    result["frames"] = frames_log
    return result


def _print_human(result: dict) -> None:
    status = result["status"]
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print("=" * 80)
    print(f"  Video        : {result['video']}")
    print(f"  video_id     : {result['video_id']}")
    print(f"  Decision     : {result.get('decision_rule', 'strict')}")
    print(f"  Status       : {color}{status}{reset}")
    print(f"  label frames : {result['label_detected_frames']}")
    print(f"  valid OCR    : {result['valid_ocr_frames']}")
    print(f"  best text    : {result['best_text'] or '(none)'}  (frame {result.get('best_frame_idx', -1)})")
    frames = result.get("frames") or []
    if frames:
        print("-" * 80)
        print(f"  {'#':<3} {'src':<14} {'conf':<6} {'valid':<6} raw_texts -> cleaned")
        for f in frames:
            raw = " | ".join(f.get("raw_texts") or []) or "(none)"
            cleaned = " | ".join(c["cleaned"] for c in (f.get("cleaned_candidates") or []) if c["cleaned"]) or "(none)"
            conf = f.get("det_conf")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "-"
            print(f"  {f['frame']:<3} {f['source']:<14} {conf_s:<6} {str(f['valid']):<6} {raw[:40]} -> {cleaned[:40]}")
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--label-class", default=DEFAULTS["label_class"])
    ap.add_argument("--sample-count", type=int, default=DEFAULTS["sample_count"])
    ap.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    ap.add_argument("--crop-pad", type=float, default=DEFAULTS["crop_pad"])
    ap.add_argument("--min-frames-detected", type=int, default=DEFAULTS["min_frames_detected"])
    ap.add_argument("--min-valid-ocr", type=int, default=DEFAULTS["min_valid_ocr"])
    ap.add_argument("--ocr", choices=["easyocr", "tesseract"], default=DEFAULTS["ocr"])
    ap.add_argument("--min-text-len", type=int, default=DEFAULTS["min_text_len"])
    ap.add_argument("--min-digit-count", type=int, default=DEFAULTS["min_digit_count"])
    ap.add_argument("--save-crops", action="store_true")
    decision = ap.add_mutually_exclusive_group()
    decision.add_argument("--strict", dest="ocr_only", action="store_false",
                          help="Require label detection AND valid OCR (production rule)")
    decision.add_argument("--ocr-only", dest="ocr_only", action="store_true",
                          help="MVP: ignore label detection, decide on OCR only (default)")
    ap.set_defaults(ocr_only=DEFAULTS["ocr_only"])
    ap.add_argument("--no-debug", dest="save_debug", action="store_false",
                    help="Do not write outputs/tracking/debug/<video_id>.json")
    ap.set_defaults(save_debug=True)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Video not found: {args.input}")

    result = check_tracking(
        video_path=args.input,
        weights=args.weights,
        sample_count=args.sample_count,
        conf=args.conf,
        label_class=args.label_class,
        crop_pad=args.crop_pad,
        min_frames_detected=args.min_frames_detected,
        min_valid_ocr=args.min_valid_ocr,
        ocr_backend=args.ocr,
        min_text_len=args.min_text_len,
        min_digit_count=args.min_digit_count,
        save_crops=args.save_crops,
        ocr_only=args.ocr_only,
        save_debug=args.save_debug,
    )
    _print_human(result)
    # Strip per-frame log from JSON block to keep stdout readable; full log
    # is in outputs/tracking/debug/<video_id>.json.
    summary = {k: v for k, v in result.items() if k != "frames"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
