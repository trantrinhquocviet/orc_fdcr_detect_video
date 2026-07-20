"""Streamlit review tool with video traceability.

Loads images + parallel YOLO .txt files, draws boxes, lets the reviewer mark
each image as Correct / Incorrect / Skip. Filters by video_id, group navigation,
saves feedback to CSV with video_id traceability.

Filename convention: <video_id>_f<NNNN>.jpg

Run:
    streamlit run scripts/review_app.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# streamlit-drawable-canvas==0.9.3 calls streamlit.elements.image.image_to_url(image, width:int, ...).
# Streamlit >=1.30 moved this to elements.lib.image_utils and changed arg 2 to a LayoutConfig.
# Shim it back so the old call site keeps working.
import streamlit.elements.image as _st_image_mod  # noqa: E402
if not hasattr(_st_image_mod, "image_to_url"):
    def _shim_image_to_url(image, width, clamp, channels, output_format, image_id):  # type: ignore[no-redef]
        from streamlit.elements.lib.image_utils import image_to_url as _new
        from streamlit.elements.lib.layout_utils import LayoutConfig
        return _new(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)
    _st_image_mod.image_to_url = _shim_image_to_url  # type: ignore[attr-defined]

ROOT = Path(os.environ.get("ORC_DATA_ROOT", str(Path(__file__).resolve().parents[1])))
FEEDBACK_DIR = ROOT / "outputs" / "review"
IMPORT_DIR = ROOT / "dataset" / "_import"  # neutral pool — frames live here until sent to a task

# Task-based dataset routing. Adding a new model = add a new entry here.
TASK_CONFIG: dict[str, dict[str, Path]] = {
    "Damage": {
        "image_dir": ROOT / "dataset" / "_raw",
        "classes_file": ROOT / "configs" / "predefined_classes.txt",
        "data_yaml": ROOT / "configs" / "data.yaml",
    },
    "Shipping Label": {
        "image_dir": ROOT / "dataset" / "shipping_label" / "_to_label",
        "classes_file": ROOT / "dataset" / "shipping_label" / "_to_label" / "predefined_classes.txt",
        "data_yaml": ROOT / "dataset" / "shipping_label" / "data.yaml",
    },
}


def _task_slug(task: str) -> str:
    return task.lower().replace(" ", "_")


# ---------- Multi-user assignment ----------
# Simulated team — replace with auth in production. Roles aren't enforced;
# anyone can claim from the unassigned pool. The "Current user" sidebar
# selectbox tags work attribution per session.
USERS = ["user_a", "user_b", "user_c", "user_d"]
ASSIGNMENT_FILENAME = "_assignments.json"

# ---------- Versioned datasets ----------
# Each import creates a new version folder under TASK_CONFIG[task]["image_dir"]
# (treated as the base). This isolates re-extracted frames from labels that
# belong to a previous extraction with the same filename.
#
# v_legacy = virtual version pointing at the base folder itself. This is how
# we surface pre-versioning datasets without any file moves.
LEGACY_VERSION = "v_legacy"
META_FILENAME = "_meta.json"


def _list_versions(task: str, include_archived: bool = False) -> list[str]:
    """Return version names for a task, newest first.

    Archived versions (meta.archived == True) are hidden by default. Pass
    include_archived=True to surface them (e.g. for an admin recovery view).

    Includes 'v_legacy' if the base folder has any images at the root level
    (i.e. data exists from before versioning was introduced).
    """
    base = TASK_CONFIG[task]["image_dir"]
    versions: list[str] = []
    if base.exists():
        subs = [c for c in base.iterdir() if c.is_dir() and c.name.startswith("v")
                and c.name != LEGACY_VERSION]
        subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for c in subs:
            if not include_archived and _load_meta(c).get("archived"):
                continue
            versions.append(c.name)
        has_root_images = any(base.glob("*.jpg")) or any(base.glob("*.png"))
        if has_root_images:
            versions.append(LEGACY_VERSION)
    return versions


def _version_dir(task: str, version: str) -> Path:
    """Resolve a version name to the actual image folder."""
    base = TASK_CONFIG[task]["image_dir"]
    if version == LEGACY_VERSION:
        return base
    return base / version


def _load_meta(version_dir: Path) -> dict:
    p = version_dir / META_FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_meta(version_dir: Path, meta: dict) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    p = version_dir / META_FILENAME
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
    )
    tmp.replace(p)


_NOTE_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _sanitize_note(note: str) -> str:
    """Filesystem-safe slug from arbitrary user input.

    Steps (in order):
      1. Strip + lowercase
      2. Normalize Vietnamese / European accents via NFKD; drop combining marks
         (e.g. "Cần Thơ" → "can tho", "đêm" → "dem")
      3. Replace any non [a-z0-9] run with a single underscore
      4. Trim leading/trailing underscores; cap at 30 chars

    Empty or all-special-chars input → empty string.
    """
    s = (note or "").strip().lower()
    if not s:
        return ""
    # NFKD splits "à" into "a" + combining accent; combining() flags the latter.
    # Vietnamese "đ" / "Đ" don't decompose this way, so handle explicitly.
    s = s.replace("đ", "d").replace("ð", "d")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # ASCII fold: anything still non-ASCII (rare after NFKD) → drop.
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = _NOTE_NON_SLUG_RE.sub("_", s).strip("_")
    return s[:30]


def _new_version_name(note: str = "") -> str:
    """Human-readable version name: v<YYYY-MM-DD>_<note>.

    If note is empty, falls back to v<YYYY-MM-DD>_<unix_suffix> so two same-day
    sends without a note still produce unique folders.
    """
    import time
    today = datetime.now().strftime("%Y-%m-%d")
    note_slug = _sanitize_note(note)
    if note_slug:
        return f"v{today}_{note_slug}"
    return f"v{today}_{int(time.time()) % 100000}"


def _assignments_path(task: str, version: str) -> Path:
    """Per-version assignments sidecar — keeps assignment state aligned with
    the dataset version. Re-import produces a new empty file."""
    return _version_dir(task, version) / ASSIGNMENT_FILENAME


def _load_assignments(task: str, version: str) -> dict[str, str]:
    """Read the per-version assignments sidecar. Returns {filename: user}."""
    p = _assignments_path(task, version)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        return {}


def _save_assignments(task: str, version: str, mapping: dict[str, str]) -> None:
    """Atomic write — temp file then rename. See multi-user race-window note
    in the previous turn's review_app.py if you hit overlap in practice."""
    p = _assignments_path(task, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8",
    )
    tmp.replace(p)


# ---------- Per-version exclude list (skip from training) ----------

EXCLUDED_FILENAME = "_excluded.json"


def _excluded_path(task: str, version: str) -> Path:
    return _version_dir(task, version) / EXCLUDED_FILENAME


def _load_excluded(task: str, version: str) -> set[str]:
    p = _excluded_path(task, version)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_excluded(task: str, version: str, names: set[str]) -> None:
    p = _excluded_path(task, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(names), indent=2), encoding="utf-8")
    tmp.replace(p)


# ---------- Archive (soft) ----------

def _is_archived(task: str, version: str) -> bool:
    if version == LEGACY_VERSION:
        return False
    return bool(_load_meta(_version_dir(task, version)).get("archived"))


def _set_archived(task: str, version: str, archived: bool) -> None:
    if version == LEGACY_VERSION:
        return  # never archive the base folder
    vdir = _version_dir(task, version)
    meta = _load_meta(vdir)
    meta["archived"] = bool(archived)
    if archived:
        meta["archived_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        meta.pop("archived_at", None)
    _save_meta(vdir, meta)


# ---------- Clean dataset (create v<date>_clean) ----------

def _clean_version(task: str, source_version: str,
                   require_review_correct: bool = False) -> tuple[str, dict]:
    """Create a new v<date>_clean version containing only:
       - frames whose .txt is non-empty (labeled), AND
       - not in the exclude list, AND
       - (optionally) marked is_correct=yes in the review CSV
    Returns (new_version_name, stats_dict).
    """
    src_dir = _version_dir(task, source_version)
    excluded = _load_excluded(task, source_version)

    review_status: dict[str, str] = {}
    if require_review_correct:
        csv_path = FEEDBACK_DIR / f"review_{_task_slug(task)}_{source_version}.csv"
        if csv_path.exists():
            try:
                with csv_path.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        # Latest row per image wins (CSV is append-only).
                        review_status[row.get("image_name", "")] = row.get("is_correct", "")
            except Exception:
                pass

    new_version = _new_version_name("clean")
    target_dir = _version_dir(task, new_version)
    if target_dir.exists():
        i = 2
        while (target_dir.parent / f"{new_version}-{i}").exists():
            i += 1
        new_version = f"{new_version}-{i}"
        target_dir = _version_dir(task, new_version)
    target_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "copied": 0,
        "skipped_unlabeled": 0,
        "skipped_excluded": 0,
        "skipped_review_no": 0,
    }
    for img in sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))):
        if img.name in excluded:
            stats["skipped_excluded"] += 1
            continue
        txt = img.with_suffix(".txt")
        if not (txt.exists() and txt.stat().st_size > 0):
            stats["skipped_unlabeled"] += 1
            continue
        if require_review_correct and review_status.get(img.name) == "no":
            stats["skipped_review_no"] += 1
            continue
        _link_or_copy(img, target_dir / img.name)
        _link_or_copy(txt, target_dir / txt.name)
        stats["copied"] += 1

    _save_meta(target_dir, {
        "task": task,
        "version": new_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_extraction_method": "clean-from-version",
        "source_version": source_version,
        "require_review_correct": require_review_correct,
        "frames_total": stats["copied"],
        "clean_stats": stats,
    })
    return new_version, stats

# Distinct BGR palette — cycled by class id. No hardcoded class meaning.
PALETTE_BGR = [
    (0, 0, 255),      # red
    (0, 200, 0),      # green
    (255, 100, 0),    # blue
    (0, 200, 255),    # yellow
    (255, 0, 200),    # magenta
    (200, 100, 255),  # purple
    (100, 200, 100),  # light green
    (50, 150, 200),   # tan
]


def color_for(cls: int) -> tuple[int, int, int]:
    return PALETTE_BGR[cls % len(PALETTE_BGR)]


def load_class_names(image_dir: Path) -> tuple[dict[int, str], str]:
    """Resolve {class_id: name} dynamically. Returns (names_dict, source_label).

    Order:
        1. <image_dir>/classes.txt              (LabelImg writes this on first save)
        2. <image_dir>/predefined_classes.txt   (LabelImg input file in this folder)
        3. <image_dir>/../predefined_classes.txt (NEW: shared file when image_dir is a versioned subfolder)
        4. <image_dir>/../data.yaml             (YOLO config)
        5. <image_dir>/../../data.yaml          (NEW: data.yaml two levels up — for v* subfolders)
    """
    # 1. classes.txt next to images
    p = image_dir / "classes.txt"
    if p.exists():
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {i: n for i, n in enumerate(names)}, "classes.txt"

    # 2. predefined_classes.txt next to images
    p = image_dir / "predefined_classes.txt"
    if p.exists():
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {i: n for i, n in enumerate(names)}, "predefined_classes.txt"

    # 3. predefined_classes.txt one level up (versioned subfolder case)
    p = image_dir.parent / "predefined_classes.txt"
    if p.exists():
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {i: n for i, n in enumerate(names)}, "../predefined_classes.txt"

    # 4 & 5. data.yaml in parent or grandparent
    for label, p in (
        ("../data.yaml", image_dir.parent / "data.yaml"),
        ("../../data.yaml", image_dir.parent.parent / "data.yaml"),
    ):
        if p.exists():
            try:
                import yaml  # ships transitively via ultralytics
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                raw = data.get("names")
                if isinstance(raw, list):
                    return {i: str(n) for i, n in enumerate(raw)}, label
                if isinstance(raw, dict):
                    return {int(k): str(v) for k, v in raw.items()}, label
            except Exception:
                pass

    return {}, "(none — falling back to cls_<id>)"


def parse_video_id(stem: str) -> tuple[str, str]:
    """Return (video_id, frame_index_str) from a filename stem like 'TTKMFB581_f0007'."""
    if "_f" in stem:
        vid, frame = stem.rsplit("_f", 1)
        return vid, f"f{frame}"
    return stem, ""


def load_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:5])
            boxes.append((cls, x, y, w, h))
        except ValueError:
            continue
    return boxes


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: dict[int, str],
) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for cls, xc, yc, w, h in boxes:
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)
        color = color_for(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = class_names.get(cls, f"cls_{cls}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def append_feedback(csv_path: Path, video_id: str, image_name: str,
                    predicted_label: str, is_correct: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "video_id", "image_name",
                             "predicted_label", "is_correct"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         video_id, image_name, predicted_label, is_correct])


def fit_to_height(img_bgr: np.ndarray, max_h: int) -> np.ndarray:
    """Downscale (only) so the displayed image height fits the viewport."""
    h, w = img_bgr.shape[:2]
    if h <= max_h:
        return img_bgr
    new_w = int(w * (max_h / h))
    return cv2.resize(img_bgr, (new_w, max_h), interpolation=cv2.INTER_AREA)


def review_tab(task: str, version: str) -> None:
    st.title("🔍 Review")
    st.caption("Kiểm tra từng ảnh — đánh dấu Đúng / Sai để lọc trước khi train.")

    folder = _version_dir(task, version)
    # Feedback CSV is per task+version so a re-import doesn't merge votes.
    csv_path = FEEDBACK_DIR / f"review_{_task_slug(task)}_{version}.csv"

    if not folder.exists():
        st.warning(
            f"Dataset folder not found for version `{version}`: `{folder}`\n\n"
            "Use the **Import frames** tab to extract a video into a new version, "
            "or pick a different version in the sidebar."
        )
        return

    class_names, class_source = load_class_names(folder)

    # Filters — inline, not sidebar
    col_filter1, col_filter2 = st.columns([1, 2])
    only_labeled = col_filter1.checkbox("Chỉ ảnh có nhãn", value=True)

    all_images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if only_labeled:
        all_images = [p for p in all_images if p.with_suffix(".txt").exists()
                      and p.with_suffix(".txt").stat().st_size > 0]
    if not all_images:
        st.warning("Không có ảnh nào. Bỏ tick 'Chỉ ảnh có nhãn' hoặc import frames trước.")
        return

    video_ids = sorted({parse_video_id(p.stem)[0] for p in all_images})
    selected_video = col_filter2.selectbox(
        "Lọc theo video", options=["(tất cả)"] + video_ids, index=0)

    if selected_video == "(tất cả)":
        images = all_images
    else:
        images = [p for p in all_images if parse_video_id(p.stem)[0] == selected_video]

    if not images:
        st.warning(f"Không có ảnh nào cho video {selected_video}")
        return

    scope_key = f"{task}::{selected_video}"
    if "idx" not in st.session_state or st.session_state.get("scope") != scope_key:
        st.session_state.idx = 0
        st.session_state.scope = scope_key

    idx = max(0, min(st.session_state.idx, len(images) - 1))
    img_path = images[idx]
    label_path = img_path.with_suffix(".txt")
    video_id, frame_idx = parse_video_id(img_path.stem)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        st.error(f"Could not read {img_path.name}")
        return

    boxes = load_boxes(label_path)
    drawn = draw_boxes(bgr, boxes, class_names)
    drawn = fit_to_height(drawn, max_h=720)
    rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    classes_in_img = sorted({class_names.get(b[0], f"cls_{b[0]}") for b in boxes})
    predicted_label = ",".join(classes_in_img) if classes_in_img else "none"

    # Header — video + frame traceability
    st.markdown(f"### Video: `{video_id}` — Frame: `{frame_idx}`")
    st.write(f"**{idx + 1} / {len(images)}** in current view  ·  "
             f"file: `{img_path.name}`  ·  boxes: **{len(boxes)}**  ·  "
             f"classes: **{predicted_label}**")

    # Action buttons placed ABOVE the image so they stay reachable.
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("⬅ Prev", use_container_width=True,
                     key=f"review_{task}_prev_{idx}") and idx > 0:
            st.session_state.idx = idx - 1
            st.rerun()
    with c2:
        if st.button("✅ Correct", type="primary", use_container_width=True,
                     key=f"review_{task}_correct_{idx}"):
            append_feedback(csv_path, video_id, img_path.name, predicted_label, "yes")
            st.session_state.idx = min(idx + 1, len(images) - 1)
            st.rerun()
    with c3:
        if st.button("❌ Incorrect", use_container_width=True,
                     key=f"review_{task}_incorrect_{idx}"):
            append_feedback(csv_path, video_id, img_path.name, predicted_label, "no")
            st.session_state.idx = min(idx + 1, len(images) - 1)
            st.rerun()
    with c4:
        if st.button("Next ➡", use_container_width=True,
                     key=f"review_{task}_next_{idx}"):
            st.session_state.idx = min(idx + 1, len(images) - 1)
            st.rerun()

    # Image rendered after the action row so buttons sit above the fold.
    st.image(rgb)

    # Group navigation — jump to first frame of next video in current view
    st.markdown("---")
    st.write("**Group navigation**")
    g1, g2 = st.columns(2)
    current_video_in_view = parse_video_id(images[idx].stem)[0]

    with g1:
        if st.button("⏮ First frame of this video", use_container_width=True,
                     key=f"review_{task}_firstframe_{idx}"):
            for i, p in enumerate(images):
                if parse_video_id(p.stem)[0] == current_video_in_view:
                    st.session_state.idx = i
                    st.rerun()

    with g2:
        if st.button("⏭ Next video", use_container_width=True,
                     key=f"review_{task}_nextvideo_{idx}"):
            for i in range(idx + 1, len(images)):
                if parse_video_id(images[i].stem)[0] != current_video_in_view:
                    st.session_state.idx = i
                    st.rerun()
                    break
            else:
                st.toast("Already at the last video in this view", icon="ℹ️")

    # Feedback log
    if csv_path.exists():
        with st.expander("Feedback so far"):
            st.code(csv_path.read_text(encoding="utf-8"))


# ---------- Run-pipeline tab ----------

UPLOAD_DIR = ROOT / "outputs" / "uploads"
VIDEO_EXTS = ("mp4", "mov", "mkv", "avi", "webm")


def _status_color(status: str) -> str:
    return {
        "AUTO_PASS": "green",
        "REVIEW_REQUIRED": "orange",
        "NO_DAMAGE": "red",
        "INVALID_INPUT": "red",
        "PASS": "green",
        "FAIL": "red",
    }.get(status, "gray")


def _resolve_video_input(uploaded, path_text: str,
                         url_text: str = "") -> Path | None:
    """Resolve any of the three input modes to a local Path.

    Priority: upload > path > url. Caller decides how to surface progress for
    URL downloads; this function delegates to _download_video_url which is
    cached by URL hash.
    """
    if uploaded is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / uploaded.name
        target.write_bytes(uploaded.getvalue())
        return target
    text = (path_text or "").strip().strip('"').strip("'")
    if text:
        return Path(text).expanduser()
    url = (url_text or "").strip()
    if url:
        # Use cache only — actual download is handled by the import_tab UI
        # so it can show progress. Returns None if not yet downloaded;
        # caller must trigger _download_video_url first.
        cache_path = _url_cache_path(url)
        if cache_path.exists():
            return cache_path
    return None


def _draw_ai_hint_overlay(bgr: np.ndarray) -> np.ndarray:
    """Center-biased translucent yellow region + caption.

    Used in the Pipeline tab when OCR returned readable text but the
    shipping_label model didn't produce a bbox — so we don't know exactly
    where the text is, but we know roughly *where to look*.
    """
    out = bgr.copy()
    h, w = out.shape[:2]
    x1, y1 = int(w * 0.20), int(h * 0.20)
    x2, y2 = int(w * 0.80), int(h * 0.80)

    # Translucent fill via alpha-blended overlay.
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), thickness=-1)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

    # Crisp border on top so the region is unambiguous.
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), thickness=3)

    label = "AI Hint: Possible Label Region"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (x1, y1 - th - 12), (x1 + tw + 10, y1), (0, 255, 255), -1)
    cv2.putText(out, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return out


def _render_result(result: dict, video_path: Path | None) -> None:
    status = result.get("final_status", "?")
    color = _status_color(status)
    st.markdown(f"### Final: :{color}[**{status}**]")
    note = result.get("confidence_note", "")
    if note:
        st.caption(note)

    qc = result.get("qc", {}) or {}
    tr = result.get("tracking", {}) or {}
    dm = result.get("damage", {}) or {}
    timings = result.get("timings", {}) or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**QC**")
        qc_status = qc.get("status", "?")
        st.markdown(f":{_status_color(qc_status)}[**{qc_status}**]")
        if qc.get("reasons"):
            st.write("Reasons: " + ", ".join(qc["reasons"]))
        st.caption(f"⏱ {timings.get('qc_sec', '—')}s")
    with c2:
        st.markdown("**Tracking**")
        valid = bool(tr.get("valid"))
        visible = bool(tr.get("visible"))
        st.markdown(
            f"valid: :{('green' if valid else 'red')}[**{valid}**]  ·  "
            f"visible: **{visible}**"
        )
        text = tr.get("text") or ""
        st.code(text or "(no readable text)", language=None)
        st.caption(f"⏱ {timings.get('tracking_sec', '—')}s")
    with c3:
        st.markdown("**Damage**")
        detected = bool(dm.get("detected"))
        st.markdown(f"detected: :{('red' if detected else 'green')}[**{detected}**]")
        st.write(f"count: **{dm.get('count', 0)}**  ·  conf: **{dm.get('confidence', 0.0)}**")
        st.caption(f"⏱ {timings.get('damage_sec', '—')}s")

    # ---- Detection visualization (bbox overlay or NO LABEL DETECTED) ----
    if video_path and video_path.exists():
        det = tr.get("detection") or {}
        det_status = det.get("status", "unknown")
        best_idx = int(det.get("best_frame_idx", -1))
        detections_count = int(det.get("detections_count", 0))
        sample_count = int(det.get("frames_sampled", 0))

        # Logging surface for the user — this matches the stdout logs from
        # run_shipping_label_detection so they can see the same info in UI.
        with st.expander(
            f"🔍 Detection · status `{det_status}` · "
            f"{detections_count}/{sample_count} frame(s) detected",
            expanded=True,
        ):
            st.code(
                f"weights      : {det.get('weights_path', '?')}\n"
                f"conf_thresh  : {det.get('conf_threshold', '?')}\n"
                f"frames       : {sample_count}\n"
                f"detections   : {detections_count}\n"
                f"confidences  : {det.get('confidences', [])}\n"
                f"best_frame   : {best_idx}\n"
                f"best_conf    : {det.get('best_confidence', '?')}\n"
                f"model classes: {det.get('model_classes', [])}",
                language="text",
            )

            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if det_status == "detected" and best_idx >= 0 and total > 0:
                # Show the frame with the highest-confidence box drawn red.
                # check_tracking samples evenly: pick that frame here too.
                # sample_frames stride: frame_index = round(i * (total-1) / (n-1))
                n = sample_count
                src_idx = (
                    int(round(best_idx * (total - 1) / max(1, n - 1)))
                    if n > 1 else total // 2
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    # Find the best-confidence detection record.
                    best_rec = next(
                        (d for d in det.get("detections", [])
                         if d.get("frame") == best_idx and d.get("detected")),
                        None,
                    )
                    if best_rec and best_rec.get("bbox_xyxy"):
                        x1, y1, x2, y2 = (int(round(v)) for v in best_rec["bbox_xyxy"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        label = f"shipping_label {best_rec['confidence']:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
                        )
                        cv2.rectangle(
                            frame, (x1, max(0, y1 - th - 8)),
                            (x1 + tw + 6, y1), (0, 0, 255), -1,
                        )
                        cv2.putText(
                            frame, label, (x1 + 3, max(th + 2, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        )
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(
                        rgb,
                        caption=(
                            f"Best detection: frame {src_idx} of {total} "
                            f"(sampled #{best_idx + 1} of {sample_count}, "
                            f"conf {det.get('best_confidence', '?')})"
                        ),
                    )
            else:
                # No detection — show mid-frame plain + a clear banner.
                if total > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb, caption=f"Frame {total // 2} of {total}")
                else:
                    cap.release()
                if det_status == "weights_missing":
                    st.error(
                        "🚫 **NO LABEL DETECTED** — `shipping_label` weights file "
                        "doesn't exist. Train the model first (Train tab) or pick "
                        "a different model in the dropdown above."
                    )
                elif det_status == "class_missing":
                    st.error(
                        "🚫 **NO LABEL DETECTED** — selected model doesn't expose "
                        "the `shipping_label` class. Pick a different model."
                    )
                elif det_status == "no_label_detected":
                    st.error(
                        f"🚫 **NO LABEL DETECTED** — model fired on 0 of "
                        f"{sample_count} sampled frame(s) at conf ≥ "
                        f"{det.get('conf_threshold', '?')}."
                    )
                else:
                    st.error(f"🚫 Detection: `{det_status}` (no usable result)")

    with st.expander("Raw JSON"):
        st.json(result)


def pipeline_tab() -> None:
    st.title("▶ Pipeline")
    st.caption("QC + OCR + damage detection.")

    # Lazy import so we can read the resolver without paying torch's cold-start
    # unless the user actually clicks Run. Used to populate the model dropdown.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pipeline_runner as _pr

    # ---- Shipping-label model selector ----
    available_models = []
    models_dir = ROOT / "models"
    if models_dir.exists():
        for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0,
                        reverse=True):
            if d.is_dir() and d.name.startswith("rl_shipping_label_"):
                wp = d / "weights" / "best.pt"
                if wp.exists():
                    available_models.append((d.name, wp))
    options = ["(auto — latest)"] + [m[0] for m in available_models]
    selected_model = st.selectbox(
        "Shipping-label model", options, index=0,
        key="pipeline_model",
        help="Loaded at Run time. 'auto' picks the most-recently-modified "
             "models/rl_shipping_label_*/weights/best.pt. If you've trained "
             "more than one version, pick a specific one to compare.",
    )
    if selected_model == options[0]:
        resolved_weights = _pr.resolve_shipping_label_weights()
    else:
        resolved_weights = next(p for n, p in available_models if n == selected_model)
    st.caption(
        f"Resolved weights: `{resolved_weights}` "
        f"({'exists' if resolved_weights.exists() else '⚠ MISSING — pipeline will return no_label_detected'})"
    )

    # Explicit keys so this tab's widgets never collide with the Import tab's
    # uploader, even if widget args change.
    upload = st.file_uploader(
        "Upload video", type=list(VIDEO_EXTS), key="pipeline_upload",
    )
    path_text = st.text_input(
        "Or paste a video path",
        placeholder=r"C:\Users\OP-LT-0496\Downloads\video.mp4",
        key="pipeline_path",
    )

    video_path = _resolve_video_input(upload, path_text)

    def _video_sig(p: Path | None) -> str | None:
        """`<abs_path>|<mtime>` — invalidates cache when same-filename uploads
        are overwritten with different bytes (mtime changes on rewrite)."""
        if p is None or not p.exists():
            return None
        return f"{p.resolve()}|{p.stat().st_mtime}"

    current_sig = _video_sig(video_path)

    # Strict cache invalidation: any change in path OR file content (mtime)
    # drops the cached result. No stale render across video swaps.
    cached_sig = st.session_state.get("pipeline_video_sig")
    if cached_sig and current_sig and cached_sig != current_sig:
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("pipeline_video", None)
        st.session_state.pop("pipeline_video_sig", None)

    if video_path is None:
        st.info("Upload a file or paste a path, then click **Run**.")

    run_clicked = st.button("Run", type="primary", disabled=video_path is None,
                            key="pipeline_run")

    if run_clicked:
        if not video_path or not video_path.exists():
            st.error(f"File not found: {video_path}")
            return
        # Clear any prior result before we start, so a failed run doesn't
        # leave stale output rendered alongside an error.
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("pipeline_video", None)
        st.session_state.pop("pipeline_video_sig", None)
        with st.spinner(f"Running model on `{video_path.name}`... please wait (~30–60s)"):
            try:
                # Lazy import — pulls in torch/ultralytics, slow on first call.
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from pipeline_runner import run_pipeline
                # NOTE: run_pipeline reads the file fresh each call. No
                # caching at any layer — qc_video, check_tracking, and infer
                # all read from disk on every call.
                result = run_pipeline(
                    video_path,
                    tracking_weights=resolved_weights,
                )
                st.session_state.pipeline_result = result
                st.session_state.pipeline_video = str(video_path)
                st.session_state.pipeline_video_sig = _video_sig(video_path)
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:  # noqa: BLE001 — surface in UI, don't crash app
                st.error(f"Pipeline failed: {type(e).__name__}: {e}")
                return
        st.success(f"Done — analyzed `{video_path.name}`.")

    # Render only when the cached result matches the current input's signature.
    result = st.session_state.get("pipeline_result")
    cached_sig = st.session_state.get("pipeline_video_sig")
    cached_path = st.session_state.get("pipeline_video")
    if result is not None and cached_sig and cached_sig == current_sig:
        _render_result(result, Path(cached_path) if cached_path else None)
    elif result is not None and current_sig is None:
        # No active input + lingering result — wipe to be safe.
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("pipeline_video", None)
        st.session_state.pop("pipeline_video_sig", None)


# ---------- Import frames tab ----------

POOL_PAGE_SIZE = 32
POOL_THUMB_W = 200
POOL_CHK_PREFIX = "import_chk_"

URL_CACHE_DIR = ROOT / "outputs" / "url_cache"
URL_DOWNLOAD_TIMEOUT_SEC = 120
URL_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _url_cache_path(url: str) -> Path:
    """Stable cache path: <sha1[:12]><ext>. Same URL → same path → cache hit."""
    import hashlib
    from urllib.parse import urlparse
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    parsed_path = urlparse(url).path
    ext = Path(parsed_path).suffix.lower()
    if ext not in URL_VIDEO_EXTS:
        ext = ".mp4"  # fall back; not all URLs expose an extension
    return URL_CACHE_DIR / f"{h}{ext}"


def _validate_video_url(url: str) -> tuple[bool, str]:
    """Cheap pre-flight: scheme + extension. Doesn't fetch headers (would
    add another round-trip and many CDNs strip Content-Type anyway).
    """
    from urllib.parse import urlparse
    try:
        p = urlparse(url)
    except Exception as e:
        return False, f"unparseable URL: {e}"
    if p.scheme not in ("http", "https"):
        return False, f"scheme must be http/https (got `{p.scheme}`)"
    if not p.netloc:
        return False, "no host in URL"
    ext = Path(p.path).suffix.lower()
    if ext and ext not in URL_VIDEO_EXTS:
        return False, (
            f"extension `{ext}` not a known video format "
            f"({', '.join(sorted(URL_VIDEO_EXTS))})"
        )
    # Empty extension allowed — some CDNs serve via /watch?v=… with no ext
    return True, ""


def _download_video_url(url: str,
                        timeout: int = URL_DOWNLOAD_TIMEOUT_SEC,
                        progress_cb=None) -> Path:
    """Download to URL_CACHE_DIR/<hash>.<ext>. Returns the cache path.

    Cache hits return immediately. Streams in 64 KB chunks; calls
    progress_cb(downloaded_bytes, total_bytes_or_None) per chunk.
    """
    import urllib.request
    target = _url_cache_path(url)
    if target.exists() and target.stat().st_size > 0:
        return target  # cache hit

    URL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "review_app/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total_str = resp.headers.get("Content-Length")
            total = int(total_str) if total_str and total_str.isdigit() else None
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return target


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link src -> dst (no extra disk on same volume), fall back to copy."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def import_tab(task: str, version: str) -> None:
    st.title("⬆ Import")
    st.caption("Tải video → trích xuất frames → chọn frames → gửi vào dataset.")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_frames import derive_video_id, validate_video_id, extract

    # ---- Apply pending pool reset BEFORE any checkbox widgets render ----
    if st.session_state.pop("_pending_pool_clear", False):
        for k in [sk for sk in st.session_state.keys() if sk.startswith(POOL_CHK_PREFIX)]:
            st.session_state[k] = False
        st.session_state.pop("import_pool_selected", None)

    # ============== Step 1: Extract video to pool ==============
    st.markdown("### Bước 1 — Tải video lên")

    upload = st.file_uploader(
        "Chọn file video", type=list(VIDEO_EXTS), key="import_upload",
    )
    with st.expander("Tùy chọn khác (đường dẫn file hoặc URL)"):
        path_text = st.text_input(
            "Đường dẫn file trên máy",
            placeholder=r"C:\Users\...\video.mp4",
            key="import_path",
        )
        url_text = st.text_input(
            "URL video (http/https)",
            placeholder="https://example.com/video.mp4",
            key="import_url",
        )
    fps = st.slider(
        "Frames per second (higher = more frames to label)",
        min_value=0.5, max_value=5.0, value=1.0, step=0.5,
        key="import_fps",
    )

    # If a URL is given but not yet cached, offer a download button.
    url_clean = (url_text or "").strip()
    if url_clean and not _url_cache_path(url_clean).exists():
        ok_url, url_err = _validate_video_url(url_clean)
        if not ok_url:
            st.error(f"Invalid URL: {url_err}")
        else:
            cache_target = _url_cache_path(url_clean)
            if st.button(
                f"⬇️ Tải về từ URL",
                use_container_width=True, key="import_url_download",
            ):
                bar = st.progress(0.0, text=f"Downloading {url_clean}…")
                status = st.empty()

                def _on_progress(downloaded: int, total: int | None) -> None:
                    if total and total > 0:
                        pct = min(downloaded / total, 1.0)
                        bar.progress(pct, text=f"Downloading… {pct:.0%}")
                    else:
                        # Unknown total — just show bytes
                        mb = downloaded / (1024 * 1024)
                        status.caption(f"Downloaded {mb:.1f} MB so far…")

                try:
                    _download_video_url(url_clean, progress_cb=_on_progress)
                    bar.progress(1.0, text="Download complete.")
                    st.success(f"✅ Cached: `{cache_target.relative_to(ROOT)}`")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    bar.empty()
                    st.error(f"Download failed: {type(e).__name__}: {e}")
    elif url_clean and _url_cache_path(url_clean).exists():
        st.caption(
            f"✓ URL cached at `{_url_cache_path(url_clean).relative_to(ROOT)}` — reusing."
        )

    video_path = _resolve_video_input(upload, path_text, url_clean)
    auto_id = ""
    if video_path is not None:
        try:
            auto_id = derive_video_id(video_path)
            validate_video_id(auto_id)
        except SystemExit:
            auto_id = ""

    if video_path is None:
        st.info("👆 Chọn file video ở trên để bắt đầu.")
    elif not auto_id:
        st.error(
            f"Tên file không hợp lệ: `{video_path.name}`. "
            "Đổi tên chỉ dùng chữ, số, dấu gạch ngang (ví dụ: `TTVN1066684755.mp4`)."
        )

    can_extract = bool(video_path is not None and auto_id)
    if st.button(
        "▶️ Trích xuất frames",
        type="primary", disabled=not can_extract,
        use_container_width=True, key="import_extract",
    ):
        IMPORT_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner("Đang trích xuất…"):
            try:
                count = extract(
                    video_path=video_path, video_id=auto_id,
                    target_fps=fps, width=None,
                    output_dir=IMPORT_DIR,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Lỗi: {type(e).__name__}: {e}")
                return
        st.session_state.last_imported_video_id = auto_id
        st.session_state[f"_extracted_fps::{auto_id}"] = float(fps)
        st.success(f"✅ Trích xuất **{count}** frames từ `{auto_id}`. Chọn frames bên dưới để thêm vào dataset.")

    st.markdown("---")

    # ============== Step 2: Per-frame selection ==============
    st.markdown("### Bước 2 — Chọn frames muốn label")

    if not IMPORT_DIR.exists():
        st.info("Chưa có frames nào. Tải video lên và trích xuất ở Bước 1.")
        return
    pool = sorted(IMPORT_DIR.glob("*.jpg")) + sorted(IMPORT_DIR.glob("*.png"))
    if not pool:
        st.info("Chưa có frames nào. Tải video lên và trích xuất ở Bước 1.")
        return

    # Group by source video so user can clearly see which frames came from where.
    by_video: dict[str, list[Path]] = {}
    for p in pool:
        vid, _ = parse_video_id(p.stem)
        by_video.setdefault(vid, []).append(p)

    # Video filter — defaults to the most-recently extracted video to prevent
    # cross-video accidental sends. Search input narrows the dropdown when
    # the pool grows past comfortable scrolling.
    search_q = st.text_input(
        "🔍 Search video_id (substring, case-insensitive)",
        value="", key="import_pool_search",
        placeholder="type to filter — e.g. `TTKMFB` or `1066`",
    ).strip().lower()
    all_vids = sorted(by_video.keys())
    if search_q:
        filtered_vids = [v for v in all_vids if search_q in v.lower()]
    else:
        filtered_vids = all_vids
    video_options = ["(all videos)"] + filtered_vids

    last_vid = st.session_state.get("last_imported_video_id")
    default_idx = (
        video_options.index(last_vid) if last_vid in video_options else 0
    )
    chosen_video = st.selectbox(
        "Show frames from",
        options=video_options,
        index=default_idx, key="import_pool_filter",
    )
    if search_q and len(filtered_vids) < len(all_vids):
        st.caption(
            f"Search matched **{len(filtered_vids)}** of {len(all_vids)} video_id(s). "
            "Clear the search box to see all."
        )
    visible = pool if chosen_video == "(all videos)" else by_video[chosen_video]

    # Selection set persists across pages and across video filter changes.
    sel_key = "import_pool_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()
    selected: set[str] = st.session_state[sel_key]

    total_visible = len(visible)
    page_count = max(1, (total_visible + POOL_PAGE_SIZE - 1) // POOL_PAGE_SIZE)
    page = st.number_input(
        f"Page (1–{page_count})",
        min_value=1, max_value=page_count, value=1, step=1,
        key="import_pool_page",
    )
    page_slice = visible[(page - 1) * POOL_PAGE_SIZE : page * POOL_PAGE_SIZE]

    # Selection breakdown by source video — visible to user for confidence.
    sel_by_video: dict[str, int] = {}
    for s in selected:
        if not Path(s).exists():
            continue
        v, _ = parse_video_id(Path(s).stem)
        sel_by_video[v] = sel_by_video.get(v, 0) + 1
    sel_summary = (
        ", ".join(f"`{v}`={n}" for v, n in sorted(sel_by_video.items()))
        if sel_by_video else "none"
    )
    st.write(
        f"Pool: **{len(pool)}** frames  ·  showing **{total_visible}**  ·  "
        f"page **{page}/{page_count}**  ·  **selected: {sel_summary}**"
    )

    # Bulk actions — write through to checkbox keys (Streamlit caches their state).
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("Select all on page", use_container_width=True,
                     key=f"import_selpage_{page}_{chosen_video}"):
            for p in page_slice:
                k = str(p)
                selected.add(k)
                st.session_state[f"{POOL_CHK_PREFIX}{k}"] = True
            st.rerun()
    with bc2:
        if st.button("Deselect page", use_container_width=True,
                     key=f"import_deselpage_{page}_{chosen_video}"):
            for p in page_slice:
                k = str(p)
                selected.discard(k)
                st.session_state[f"{POOL_CHK_PREFIX}{k}"] = False
            st.rerun()
    with bc3:
        if st.button("Clear all selection", use_container_width=True,
                     key="import_clearall"):
            for k in [sk for sk in st.session_state.keys() if sk.startswith(POOL_CHK_PREFIX)]:
                st.session_state[k] = False
            selected.clear()
            st.rerun()

    # Frame grid — checkbox per frame, video_id stamped on each so user can
    # never accidentally include a frame from another video.
    cols_per_row = 4
    for row_start in range(0, len(page_slice), cols_per_row):
        row = page_slice[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, row):
            with col:
                rgb = _load_thumbnail(str(p), p.stat().st_mtime, POOL_THUMB_W,
                                      square=True)
                st.image(rgb, use_container_width=True)
                vid, frame_part = parse_video_id(p.stem)
                k = str(p)
                checked = st.checkbox(
                    f"`{vid}` · {frame_part}",
                    value=(k in selected),
                    key=f"{POOL_CHK_PREFIX}{k}",
                )
                if checked and k not in selected:
                    selected.add(k)
                elif not checked and k in selected:
                    selected.discard(k)

    st.markdown("---")

    # ============== Step 3: Send to dataset ==============
    st.markdown("### Bước 3 — Thêm vào dataset")

    # Auto-use current task — no need for user to pick
    target_task = task
    final_paths = sorted({Path(s) for s in selected if Path(s).exists()})
    append_target_version = version

    _append_send_block(
        target_task=target_task,
        target_version=append_target_version,
        final_paths=final_paths,
    )

    with st.expander("⚙️ Tạo version mới (nâng cao)"):
        st.caption("Dùng khi muốn tách riêng batch mới khỏi data cũ.")
        _create_new_version_block(
            target_task=target_task,
            final_paths=final_paths,
        )


def _append_send_block(target_task: str, target_version: str,
                       final_paths: list[Path]) -> None:
    """Append-to-current behavior. Does NOT touch session_state, does NOT switch
    sidebar version, does NOT clear caches that hold user context.
    """
    target_dir = _version_dir(target_task, target_version)
    final_count = len(final_paths)
    if not st.button(
        f"✅ Thêm {final_count} frames vào dataset" if final_count > 0 else "Chưa chọn frame nào",
        type="primary", use_container_width=True,
        disabled=final_count == 0,
        key=f"import_append_{target_task}_{target_version}",
    ):
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    ok, skipped = 0, 0
    for src in final_paths:
        dst = target_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        _link_or_copy(src, dst)
        ok += 1

    # Cache_data invalidation only — does NOT touch label_active::*, queue::*,
    # or filter widget keys, so user's labeling context is preserved.
    st.cache_data.clear()

    new_total = (
        sum(1 for _ in target_dir.glob("*.jpg"))
        + sum(1 for _ in target_dir.glob("*.png"))
    )
    msg = f"✅ Đã thêm **{ok}** frames vào dataset."
    if skipped:
        msg += f" ({skipped} frame đã có, bỏ qua)"
    st.success(msg)
    st.caption(f"Dataset `{target_version}` hiện có **{new_total}** ảnh.")


def _create_new_version_block(target_task: str, final_paths: list[Path]) -> None:
    """Original create-new-version behavior. Switches sidebar version,
    seeds Inspect-tab filters for the new version. Use intentionally.
    """
    final_count = len(final_paths)

    note_raw = st.text_input(
        "Version note (short, e.g. `clean`, `fix fps`, `Cần Thơ batch`)",
        value="", key="import_send_note",
        max_chars=50, placeholder="leave empty for date-only auto-name",
        help=(
            "Used as part of the version folder name. Will be normalized: "
            "lowercased, Vietnamese/European accents stripped, spaces and "
            "special chars → underscore, capped at 30 chars."
        ),
    )
    sanitized_note = _sanitize_note(note_raw)
    today = datetime.now().strftime("%Y-%m-%d")
    preview_folder = (
        f"v{today}_{sanitized_note}" if sanitized_note
        else f"v{today}_<auto-suffix>"
    )
    base_rel = TASK_CONFIG[target_task]["image_dir"].relative_to(ROOT)

    # Show the sanitized version so the user sees what name will actually
    # appear on disk before they commit.
    if note_raw.strip() and note_raw.strip() != sanitized_note:
        st.caption(
            f"Sanitized: `{note_raw.strip()}` → "
            f"`{sanitized_note or '(empty after sanitize → auto-suffix will be used)'}`"
        )
    st.caption(f"Will create: `{base_rel}/{preview_folder}/`")

    if not st.button(
        f"🚀 Create new version with {final_count} frame(s) in {target_task}",
        type="primary", use_container_width=True,
        disabled=final_count == 0,
        key=f"import_send_new_{target_task}",
    ):
        return

    new_version = _new_version_name(note_raw)
    target_dir = _version_dir(target_task, new_version)
    if target_dir.exists():
        # Same-day note collision — append a counter.
        i = 2
        while (target_dir.parent / f"{new_version}-{i}").exists():
            i += 1
        new_version = f"{new_version}-{i}"
        target_dir = _version_dir(target_task, new_version)
    target_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0
    videos_in_send: dict[str, int] = {}
    for src in final_paths:
        dst = target_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        _link_or_copy(src, dst)
        ok += 1
        vid, _ = parse_video_id(src.stem)
        videos_in_send[vid] = videos_in_send.get(vid, 0) + 1

    fps_by_video = {
        vid: st.session_state.get(f"_extracted_fps::{vid}")
        for vid in videos_in_send
    }
    unique_fps = {f for f in fps_by_video.values() if f is not None}
    single_fps = next(iter(unique_fps)) if len(unique_fps) == 1 else None

    _save_meta(target_dir, {
        "task": target_task,
        "version": new_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_extraction_method": "opencv-step",
        "frames_total": ok,
        "frames_by_video": videos_in_send,
        "fps": single_fps,
        "fps_by_video": fps_by_video,
        "source_pool": str(IMPORT_DIR.relative_to(ROOT)),
    })

    v_filter = (
        next(iter(videos_in_send.keys())) if len(videos_in_send) == 1 else "(all)"
    )
    for u in USERS:
        st.session_state[f"label_status_{target_task}_{new_version}_{u}"] = "Unlabeled"
        st.session_state[f"label_vid_{target_task}_{new_version}_{u}"] = v_filter
        st.session_state[f"label_assignment_{target_task}_{new_version}_{u}"] = "My work"
        st.session_state.pop(f"label_active::{target_task}::{new_version}::{u}", None)
        st.session_state.pop(f"selected_images::{target_task}::{new_version}::{u}", None)

    st.session_state[f"_pending_version_switch::{target_task}"] = new_version
    st.cache_data.clear()
    st.success(
        f"✅ Created `{new_version}` in **{target_task}** with {ok} frame(s) "
        f"({skipped} already present, skipped). "
        "Sidebar will switch to this version automatically."
    )
    st.rerun()


# ---------- In-browser label tab ----------

DISPLAY_MAX_W = 900  # canvas width cap; original-image coords reconstructed via scale.
GRID_PAGE_SIZE = 50
GRID_THUMB_W = 240


@st.cache_data(show_spinner=False)
def _image_md5(path_str: str, mtime: float) -> str:
    """Cached MD5 of an image file. mtime in cache key invalidates on overwrite.
    Used by Safe-mode carry-forward — only computed for filename-matched pairs.
    """
    import hashlib
    h = hashlib.md5()
    with open(path_str, "rb") as f:
        # 1 MB chunks — keeps memory bounded on large frames.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _load_thumbnail(path_str: str, mtime: float, max_w: int,
                    square: bool = False) -> np.ndarray:
    """Load + resize thumbnail (RGB). Cache invalidated when file mtime changes.

    square=True center-crops to a square then scales to max_w × max_w.
    Equivalent to CSS `object-fit: cover` — fills without distortion, cropping
    only the longest axis. Used by the Inspect/Pool grids so rows align.
    """
    bgr = cv2.imread(path_str)
    if bgr is None:
        # Empty placeholder. Square if requested so the grid stays aligned.
        side = max_w if square else max_w * 9 // 16
        return np.zeros((side, max_w, 3), dtype=np.uint8)
    h0, w0 = bgr.shape[:2]
    if square:
        side = min(h0, w0)
        y0, x0 = (h0 - side) // 2, (w0 - side) // 2
        cropped = bgr[y0:y0 + side, x0:x0 + side]
        thumb = cv2.resize(cropped, (max_w, max_w), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    if w0 <= max_w:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    thumb_h = int(h0 * (max_w / w0))
    thumb = cv2.resize(bgr, (max_w, thumb_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)


def _is_labeled(img_path: Path) -> bool:
    txt = img_path.with_suffix(".txt")
    return txt.exists() and txt.stat().st_size > 0


# ---------- Semi-auto labeling (model suggestions) ----------
# When the toggle is ON in the Inspect tab, clicking a thumbnail computes
# bbox predictions from the latest trained model. Predictions are shown on
# the preview as yellow outlines. Clicking "Open in LabelImg" pre-fills
# <frame>.txt with those predictions ONLY IF the file doesn't already have
# user labels. Existing labels are never overwritten silently.

SUGGESTION_CONF_THRESHOLD = 0.3  # per spec


@st.cache_data(show_spinner=False)
def _predict_boxes_for_image(img_path_str: str, mtime: float,
                             weights_path_str: str,
                             conf_threshold: float) -> list[dict]:
    """Run YOLO on a single image. Returns YOLO-normalized boxes.

    Cache key: (path, mtime, weights, threshold). Re-extracting the image
    or training a new model invalidates the cache for that frame.
    """
    if not Path(weights_path_str).exists():
        return []
    if not Path(img_path_str).exists():
        return []
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path_str)
        img = cv2.imread(img_path_str)
        if img is None:
            return []
        H, W = img.shape[:2]
        results = model.predict(img, conf=conf_threshold, verbose=False)
    except Exception:
        return []

    out: list[dict] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(cls_arr)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            out.append({
                "class_id": int(cls_arr[i]),
                "cx": round(cx, 6), "cy": round(cy, 6),
                "w": round(w, 6), "h": round(h, 6),
                "confidence": round(float(confs[i]), 4),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    return out


def _prefill_label_from_predictions(img_path: Path,
                                    predictions: list[dict],
                                    force: bool = False) -> int:
    """Write predictions to <frame>.txt in YOLO format.

    Returns:
       N>0  — wrote N lines successfully
       0    — nothing to write (empty predictions)
       -1   — skipped (existing non-empty .txt and force=False)
    """
    if not predictions:
        return 0
    txt = img_path.with_suffix(".txt")
    if txt.exists() and txt.stat().st_size > 0 and not force:
        return -1
    lines = [
        f"{p['class_id']} {p['cx']:.6f} {p['cy']:.6f} {p['w']:.6f} {p['h']:.6f}"
        for p in predictions
    ]
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _draw_prediction_overlay(rgb: np.ndarray,
                             predictions: list[dict]) -> np.ndarray:
    """Draw yellow outlines for model predictions on a copy of the image."""
    out = rgb.copy()
    H, W = out.shape[:2]
    for p in predictions:
        cx, cy, w, h = p["cx"], p["cy"], p["w"], p["h"]
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        # Yellow in RGB (cv2.rectangle on RGB array works).
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 3)
        label = f"suggested {p['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 6, y1), (255, 200, 0), -1)
        cv2.putText(out, label, (x1 + 3, max(th + 2, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


# ---------- Auto-OCR on labeled bounding boxes ----------
# When a frame's .txt is newer than its .ocr.json sidecar, crop each labeled
# box and run easyocr. Stores results next to the image. The cached
# st.cache_resource Reader avoids a 3-5s cold-start per render.

OCR_MAX_PER_RENDER = 5  # cap synchronous OCR per Streamlit rerun


@st.cache_resource(show_spinner=False)
def _get_easyocr_reader():
    """One Reader per Streamlit session. ~64 MB English model, CPU-only."""
    import easyocr  # noqa: WPS433
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _ocr_sidecar_path(img_path: Path) -> Path:
    return img_path.with_suffix(".ocr.json")


def _needs_ocr(img_path: Path) -> bool:
    """True if labeled and OCR is missing / older than the label."""
    if not _is_labeled(img_path):
        return False
    txt = img_path.with_suffix(".txt")
    side = _ocr_sidecar_path(img_path)
    if not side.exists():
        return True
    return txt.stat().st_mtime > side.stat().st_mtime


def _load_ocr(img_path: Path) -> list[dict]:
    p = _ocr_sidecar_path(img_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data.get("boxes", []))
    except Exception:
        return []


def _ocr_best_rotation(crop: np.ndarray, reader) -> tuple[str, float, int]:
    """Try 0°/90°/180°/270° rotations; pick the one with highest OCR confidence.

    Returns (text, confidence, rotation_degrees). Empty crop / unreadable
    text → ("", 0.0, 0). Cost: ~4× single-OCR time per box, but the result
    is cached in <frame>.ocr.json so this only runs once per labeled box.
    """
    if crop.size == 0:
        return "", 0.0, 0
    rotation_ops = {
        0: lambda x: x,
        90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        180: lambda x: cv2.rotate(x, cv2.ROTATE_180),
        270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    best_text, best_conf, best_rot = "", 0.0, 0
    for rot, op in rotation_ops.items():
        try:
            ocr_out = reader.readtext(op(crop), detail=1, paragraph=False)
        except Exception:
            continue
        if not ocr_out:
            continue
        # Score = mean confidence weighted by text length (longer matches win
        # ties — typical on shipping labels with a single tracking code).
        confs = [float(c) for _, _, c in ocr_out]
        texts = [t for _, t, _ in ocr_out]
        text = " ".join(texts).strip()
        conf = max(confs, default=0.0)
        # Tie-break: prefer the rotation with both higher conf AND non-empty text.
        if conf > best_conf and text:
            best_text, best_conf, best_rot = text, conf, rot
    return best_text, best_conf, best_rot


def _run_ocr_for_image(img_path: Path) -> int:
    """Crop each labeled box, try 4 rotations, OCR. Writes <frame>.ocr.json.

    Schema (per box):
        auto_text / auto_confidence / auto_rotation_deg  — overwritten each run
        corrected_text / corrected_at / corrected_by      — preserved if set

    Re-running OCR (e.g. after relabeling) refreshes auto_* but never wipes a
    user correction. Use _save_correction() to set the corrected_* fields.
    """
    txt = img_path.with_suffix(".txt")
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return 0
    H, W = bgr.shape[:2]
    boxes = load_boxes(txt)
    reader = _get_easyocr_reader()

    # Preserve any prior corrections — they're ground truth.
    prior_by_index: dict[int, dict] = {}
    side = _ocr_sidecar_path(img_path)
    if side.exists():
        try:
            for b in json.loads(side.read_text(encoding="utf-8")).get("boxes", []):
                if b.get("corrected_text"):
                    prior_by_index[int(b["box_index"])] = b
        except Exception:
            pass

    results = []
    for i, (cls, cx, cy, w, h) in enumerate(boxes):
        x1 = max(0, int((cx - w / 2) * W))
        y1 = max(0, int((cy - h / 2) * H))
        x2 = min(W, int((cx + w / 2) * W))
        y2 = min(H, int((cy + h / 2) * H))
        crop = bgr[y1:y2, x1:x2]
        text, conf, rot = _ocr_best_rotation(crop, reader)
        entry = {
            "box_index": i,
            "class_id": int(cls),
            "bbox_norm": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
            "auto_text": text,
            "auto_confidence": round(conf, 3),
            "auto_rotation_deg": rot,
            "corrected_text": None,
            "corrected_at": None,
            "corrected_by": None,
        }
        # Carry forward any prior correction for this box index.
        if i in prior_by_index:
            prev = prior_by_index[i]
            entry["corrected_text"] = prev.get("corrected_text")
            entry["corrected_at"] = prev.get("corrected_at")
            entry["corrected_by"] = prev.get("corrected_by")
        results.append(entry)

    payload = {
        "image": img_path.name,
        "ocr_at": datetime.now().isoformat(timespec="seconds"),
        "boxes": results,
    }
    side.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return len(results)


def _resolved_ocr_text(box: dict) -> tuple[str, bool]:
    """Returns (ground_truth_text, is_user_corrected)."""
    corrected = (box.get("corrected_text") or "").strip()
    if corrected:
        return corrected, True
    return (box.get("auto_text") or "").strip(), False


def _save_correction(img_path: Path, box_index: int, text: str, user: str) -> bool:
    """Write a user correction to the sidecar. Empty text → clear correction.
    Returns True on successful update.
    """
    side = _ocr_sidecar_path(img_path)
    if not side.exists():
        return False
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except Exception:
        return False
    cleaned = (text or "").strip()
    found = False
    for b in data.get("boxes", []):
        if int(b.get("box_index", -1)) == box_index:
            if cleaned:
                b["corrected_text"] = cleaned
                b["corrected_at"] = datetime.now().isoformat(timespec="seconds")
                b["corrected_by"] = user
            else:
                b["corrected_text"] = None
                b["corrected_at"] = None
                b["corrected_by"] = None
            found = True
            break
    if not found:
        return False
    side.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return True


def _validate_ocr_text(task: str, text: str) -> tuple[bool, str]:
    """Per-task plausibility rule. Returns (valid, reason_if_invalid).

    Used to flag suspect auto-OCR — guides reviewers to frames that need
    correction. Doesn't block anything; purely advisory.
    """
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if task == "Shipping Label":
        cleaned = re.sub(r"[^A-Z0-9]", "", t.upper())
        if len(cleaned) < 8:
            return False, f"length {len(cleaned)} <8 (tracking codes are 8+ alphanum)"
        if len(cleaned) > 20:
            return False, f"length {len(cleaned)} >20 (tracking codes are ≤20)"
        digits = sum(c.isdigit() for c in cleaned)
        if digits < 6:
            return False, f"only {digits} digits (tracking codes are digit-heavy)"
        return True, ""
    return True, ""


def _scan_and_run_ocr(version_dir: Path,
                      max_per_render: int = OCR_MAX_PER_RENDER) -> int:
    """Process up to N stale-OCR frames synchronously. Returns frame count.

    Synchronous on purpose — keeps state simple and matches the user's
    save-then-glance flow. easyocr is fast on small label crops (~0.3s ea).
    """
    candidates = sorted(
        list(version_dir.glob("*.jpg")) + list(version_dir.glob("*.png"))
    )
    pending = [p for p in candidates if _needs_ocr(p)]
    if not pending:
        return 0
    processed = 0
    for p in pending[:max_per_render]:
        try:
            _run_ocr_for_image(p)
            processed += 1
        except Exception:
            # Best-effort — never crash the UI on an OCR hiccup.
            continue
    return processed


def _preview_panel(
    task: str,
    img_path: Path,
    nav_list: list[Path],
    class_names: dict[int, str],
    suggestion_weights: Path | None = None,
    suggestions_on: bool = False,
) -> None:
    """Read-only inspector for one image. Renders existing YOLO boxes as an
    overlay (using draw_boxes from the Review tab). Editing is delegated to
    LabelImg — see the guidance section in label_tab.

    suggestion_weights + suggestions_on: when ON and the frame has NO existing
    label, render predicted boxes (in yellow) over the preview. Predictions are
    NOT written to disk by this function — only by the LabelImg-launch handler.
    """
    label_path = img_path.with_suffix(".txt")

    paths = [str(p) for p in nav_list]
    try:
        idx = paths.index(str(img_path))
    except ValueError:
        idx = 0

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        st.error(f"Could not read {img_path.name}")
        return

    boxes = load_boxes(label_path) if _is_labeled(img_path) else []
    drawn = draw_boxes(bgr, boxes, class_names)
    drawn = fit_to_height(drawn, max_h=720)
    rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    # Semi-auto labeling overlay: only when suggestions are on AND the frame
    # is unlabeled AND a trained model exists. Visual hint only — disk write
    # happens at LabelImg-launch time.
    suggestions: list[dict] = []
    if (suggestions_on and not _is_labeled(img_path)
            and suggestion_weights and suggestion_weights.exists()):
        try:
            suggestions = _predict_boxes_for_image(
                str(img_path),
                img_path.stat().st_mtime,
                str(suggestion_weights),
                SUGGESTION_CONF_THRESHOLD,
            )
        except Exception:
            suggestions = []
        if suggestions:
            rgb = _draw_prediction_overlay(rgb, suggestions)

    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        with h1:
            badge = "🟢" if _is_labeled(img_path) else "🟡"
            classes_in_img = sorted({class_names.get(b[0], f"cls_{b[0]}") for b in boxes})
            class_summary = ", ".join(classes_in_img) if classes_in_img else "no labels"
            st.markdown(
                f"### {badge} `{img_path.name}`  ·  {idx + 1} of {len(nav_list)}  "
                f"·  {len(boxes)} box(es)  ·  {class_summary}"
            )
        with h2:
            if st.button("✕ Close", use_container_width=True,
                         key=f"preview_close_{task}"):
                st.session_state[f"_pending_active::{task}"] = ""  # clear active
                st.session_state.pop(f"selected_images::{task}", None)  # drop snapshot
                st.rerun()

        st.image(rgb, use_container_width=True)

        if suggestions:
            confs = [s["confidence"] for s in suggestions]
            st.caption(
                f"💡 **{len(suggestions)} model suggestion(s)** "
                f"(conf {min(confs):.2f}–{max(confs):.2f}, threshold "
                f"{SUGGESTION_CONF_THRESHOLD}). Yellow boxes above. Click "
                "**🚀 Open in LabelImg** to write them to `.txt` and adjust."
            )

        if _is_labeled(img_path):
            ocr_boxes = _load_ocr(img_path)
            if ocr_boxes:
                st.markdown("**🧠 OCR — auto + your corrections (ground truth)**")
                # Resolve which user we attribute the correction to (the
                # "Current user" sidebar selector is the source of truth).
                current_user = st.session_state.get("current_user", USERS[0])
                for b in ocr_boxes:
                    bi = int(b.get("box_index", 0))
                    auto_text = (b.get("auto_text") or "").strip()
                    corrected_text = (b.get("corrected_text") or "").strip()
                    cls = b.get("class_id", "?")
                    conf = b.get("auto_confidence", 0.0)
                    rot = b.get("auto_rotation_deg", 0)
                    rot_note = f" · rotated `{rot}°`" if rot else ""

                    # Validation badge — purely advisory, helps reviewer triage.
                    resolved, is_corrected = _resolved_ocr_text(b)
                    valid, reason = _validate_ocr_text(task, resolved)
                    val_badge = "✅ valid" if valid else f"⚠ {reason}"

                    with st.container(border=True):
                        rc1, rc2 = st.columns([3, 2])
                        with rc1:
                            head = (
                                f"**Box `#{bi + 1}`** · class `{cls}` · "
                                f"auto conf `{conf}`{rot_note} · {val_badge}"
                            )
                            st.markdown(head)
                            st.caption(f"🤖 auto: `{auto_text or '(empty)'}`")
                            if is_corrected:
                                cb = b.get("corrected_by") or "?"
                                ca = b.get("corrected_at") or "?"
                                st.caption(f"✏ corrected by `{cb}` at `{ca}`")
                        with rc2:
                            edit_key = f"ocr_edit_{img_path.name}_{bi}"
                            new_text = st.text_input(
                                "Ground truth (editable)",
                                value=corrected_text or auto_text,
                                key=edit_key,
                                label_visibility="visible",
                            )
                            save_label = (
                                "💾 Save correction" if not is_corrected
                                else "🔄 Update correction"
                            )
                            if st.button(
                                save_label, use_container_width=True,
                                key=f"ocr_save_{img_path.name}_{bi}",
                            ):
                                if _save_correction(img_path, bi, new_text, current_user):
                                    st.toast(f"Saved correction for box #{bi + 1}.")
                                else:
                                    st.error("Could not save (sidecar missing).")
                                st.rerun()
                            if is_corrected:
                                if st.button(
                                    "♻️ Clear correction (revert to auto)",
                                    use_container_width=True,
                                    key=f"ocr_clear_{img_path.name}_{bi}",
                                ):
                                    _save_correction(img_path, bi, "", current_user)
                                    st.toast(f"Cleared correction for box #{bi + 1}.")
                                    st.rerun()
            with st.expander(f"Label file content: {label_path.name}"):
                st.code(label_path.read_text(encoding="utf-8") or "(empty)")
            ocr_path = _ocr_sidecar_path(img_path)
            if ocr_path.exists():
                with st.expander(f"OCR sidecar: {ocr_path.name}"):
                    st.code(ocr_path.read_text(encoding="utf-8"))
        else:
            st.caption("No label file — click **Open in LabelImg** below to create one.")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("⬅ Prev", use_container_width=True, disabled=idx == 0,
                         key=f"preview_prev_{task}_{img_path.name}"):
                st.session_state[f"_pending_active::{task}"] = str(nav_list[idx - 1])
                st.rerun()
        with b2:
            at_end = idx >= len(nav_list) - 1
            if st.button("Next ➡", use_container_width=True, disabled=at_end,
                         key=f"preview_next_{task}_{img_path.name}"):
                st.session_state[f"_pending_active::{task}"] = str(nav_list[idx + 1])
                st.rerun()


def _labelimg_command(folder: Path, classes_file: Path,
                      image_path: Path | None = None) -> str:
    """Render the bash equivalent of what _launch_labelimg runs.

    If image_path is given, LabelImg's first arg is the image file; LabelImg
    auto-loads that frame and the side panel still shows the rest of the folder.
    Otherwise, first arg is the folder (LabelImg opens at the first image).
    """
    first = str(image_path) if image_path is not None else str(folder)
    return (
        f'.venv/Scripts/python.exe .venv/Lib/site-packages/labelImg/labelImg.py '
        f'"{first}" "{classes_file}" "{folder}"'
    )


def _launch_labelimg(folder: Path, classes_file: Path,
                     image_path: Path | None = None) -> tuple[bool, str]:
    """Spawn LabelImg as a non-blocking subprocess. Returns (ok, message).

    When image_path is provided, LabelImg opens at THAT specific image. cwd is
    set to the image's parent directory so any relative-path file dialogs land
    in the dataset folder. Save dir is always the dataset folder regardless.
    """
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    labelimg_script = ROOT / ".venv" / "Lib" / "site-packages" / "labelImg" / "labelImg.py"

    for label, p in (
        ("venv python", venv_python),
        ("labelImg.py", labelimg_script),
        ("image folder", folder),
        ("classes file", classes_file),
    ):
        if not p.exists():
            return False, f"Cannot launch — {label} not found: {p}"

    if image_path is not None:
        if not image_path.exists():
            return False, f"Selected image no longer exists: {image_path}"
        first_arg = str(image_path)
        cwd = str(image_path.parent)
    else:
        first_arg = str(folder)
        cwd = str(ROOT)

    # Save dir (3rd arg) stays the dataset folder — LabelImg writes <name>.txt
    # next to the image regardless of which frame was opened first.
    cmd = [str(venv_python), str(labelimg_script),
           first_arg, str(classes_file), str(folder)]
    try:
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(cmd, cwd=cwd, creationflags=flags, close_fds=True)
        target = Path(first_arg).name
        return True, f"Launched LabelImg → {target}"
    except Exception as e:  # noqa: BLE001 — surface in UI
        return False, f"{type(e).__name__}: {e}"


def label_tab(task: str, version: str) -> None:
    st.title("🏷 Label")
    st.caption("Xem frames + nhãn YOLO. Vẽ bounding box chi tiết bằng LabelImg.")

    cfg = TASK_CONFIG[task]
    folder = _version_dir(task, version)
    classes_file = cfg["classes_file"]
    if not folder.exists():
        st.error(
            f"Dataset folder not found for version `{version}`: {folder}\n\n"
            "Use the **Import frames** tab to extract a video into a new version, "
            "or pick a different version in the sidebar."
        )
        return

    class_names, _ = load_class_names(folder)
    if not class_names:
        st.error(
            f"No class file resolved for task '{task}' / version `{version}`. "
            f"Place a classes.txt or predefined_classes.txt in {folder} "
            f"or in its parent ({folder.parent})."
        )
        return

    all_images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not all_images:
        st.warning(f"No images in this dataset version (`{version}`).")
        return

    # ---- Auto-OCR pass on labeled boxes. Runs synchronously each render,
    # capped to keep the UI responsive. easyocr Reader is cached, so the
    # first render of a session pays ~3-5s of cold-start; subsequent reruns
    # only pay per-frame inference (~0.3s per box on CPU). ----
    ocr_processed = _scan_and_run_ocr(folder, max_per_render=OCR_MAX_PER_RENDER)
    if ocr_processed > 0:
        st.toast(f"🧠 OCR'd {ocr_processed} new label(s)")

    # ---- Active-version banner + summary panel ----
    meta = _load_meta(folder)
    labeled_total = sum(1 for p in all_images if _is_labeled(p))
    created_at = meta.get("created_at", "—")

    with st.container(border=True):
        st.markdown(
            f"### 🔥 ACTIVE VERSION: `{version}`  ·  task: `{task}`"
        )
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Images", len(all_images))
        sm2.metric("Labeled", labeled_total)
        sm3.metric("Unlabeled", len(all_images) - labeled_total)
        sm4.metric("Created", created_at)
        # Switch-warning: detect if user just changed version while having
        # active work in the previous one. Informational only — state is
        # preserved per-version, so switching back restores everything.
        prev_active_v_key = f"_prev_active_version::{task}"
        prev_active_v = st.session_state.get(prev_active_v_key)
        if prev_active_v is not None and prev_active_v != version:
            # Did the previous version have any in-flight work?
            user_for_check = st.session_state.get("current_user", USERS[0])
            prev_active = st.session_state.get(
                f"label_active::{task}::{prev_active_v}::{user_for_check}"
            )
            prev_queue = st.session_state.get(
                f"label_queue::{task}::{prev_active_v}::{user_for_check}", []
            )
            if prev_active or prev_queue:
                st.warning(
                    f"⚠ You had active work in `{prev_active_v}` "
                    f"({len(prev_queue)} queued). It's preserved there — "
                    "switch back in the sidebar to resume. Nothing was lost."
                )
        st.session_state[prev_active_v_key] = version

    # ---- Multi-user + multi-version scoping. Every stateful key includes
    # both user and version so switching either dimension preserves
    # independent queues / active selections / filter prefs. ----
    user = st.session_state.get("current_user", USERS[0])
    assignments = _load_assignments(task, version)
    excluded: set[str] = _load_excluded(task, version)

    active_key = f"label_active::{task}::{version}::{user}"
    queue_key = f"label_queue::{task}::{version}::{user}"
    snap_key = f"selected_images::{task}::{version}::{user}"
    prev_vid_key = f"_prev_label_vid::{task}::{version}::{user}"
    status_w = f"label_status_{task}_{version}_{user}"
    vid_w = f"label_vid_{task}_{version}_{user}"
    asg_w = f"label_assignment_{task}_{version}_{user}"
    page_w = f"label_page_{task}_{version}_{user}"
    queue_chk_prefix = f"queue_chk_{task}_{version}_{user}_"

    # Apply pending active-path switch BEFORE any widget is instantiated.
    pending = st.session_state.pop(f"_pending_active::{task}::{version}::{user}", None)
    if pending is not None:
        if pending == "":
            st.session_state.pop(active_key, None)
        else:
            st.session_state[active_key] = pending

    # Apply pending filter reset BEFORE the Status/Video widgets are created.
    if st.session_state.pop(f"_pending_filter_reset::{task}::{version}::{user}", False):
        st.session_state[status_w] = "All"
        st.session_state[vid_w] = "(all)"
        st.session_state[asg_w] = "My work"

    # If the user changed the video filter to a DIFFERENT video, drop the
    # active image — otherwise the LabelImg button could launch a frame from
    # the previous filter context (the bug that prompted this whole fix).
    prev_vid = st.session_state.get(prev_vid_key)
    current_vid = st.session_state.get(vid_w, "(all)")
    if prev_vid is not None and prev_vid != current_vid:
        st.session_state.pop(active_key, None)
        st.session_state.pop(snap_key, None)
    st.session_state[prev_vid_key] = current_vid

    # ---- Resolve active image path. Single source of truth: full path string.
    # No index, no list-position. Used by the LabelImg button below + preview.
    active_str = st.session_state.get(active_key)
    active_path: Path | None = None
    if active_str:
        cand = Path(active_str)
        if cand.exists():
            active_path = cand
        else:
            st.session_state.pop(active_key, None)

    # ---- Resolve queue (multi-select). Path-based, ordered, persisted. ----
    if queue_key not in st.session_state:
        st.session_state[queue_key] = []
    queue: list[str] = st.session_state[queue_key]
    # Drop stale entries (files moved/deleted since they were queued).
    queue[:] = [p for p in queue if Path(p).exists()]

    # First unlabeled in queue order = next thing LabelImg should open.
    next_in_queue: Path | None = None
    for p_str in queue:
        cand = Path(p_str)
        if not _is_labeled(cand):
            next_in_queue = cand
            break

    # Launch target priority: queue > active > nothing.
    if next_in_queue is not None:
        launch_target: Path | None = next_in_queue
        launch_mode = "queue"
    elif queue:
        # Whole queue is already labeled — fall back to first item (review path)
        launch_target = Path(queue[0])
        launch_mode = "queue (all labeled)"
    elif active_path is not None:
        launch_target = active_path
        launch_mode = "selected"
    else:
        launch_target = None
        launch_mode = "none"

    # ---- Per-task labeling guideline (prominent banner) ----
    # Shipping Label specifically: box ONLY the alphanumeric tracking text,
    # not the barcode above it. Misframing produces garbage OCR.
    if task == "Shipping Label":
        st.warning(
            "📌 **Labeling rule:** box the **tracking-code text** only "
            "(e.g. `TTVN1064832858`). **Do NOT box the barcode above it** — "
            "OCR can't read barcodes; it produces garbage and the model learns "
            "the wrong pattern. Tight box around the readable alphanumeric line only."
        )

    # ---- Semi-auto labeling toggle (resolved once per render) ----
    # Resolves the latest trained shipping_label model. Predictions render as
    # yellow boxes on the preview. Pre-fill happens at LabelImg-launch time.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pipeline_runner as _pr_for_suggestions
    suggestion_weights = (
        _pr_for_suggestions.resolve_shipping_label_weights()
        if task == "Shipping Label" else None
    )
    suggestions_on = st.checkbox(
        "💡 Use model suggestions (semi-auto labeling)",
        value=False,
        key=f"label_suggestions_{task}_{version}_{user}",
        help=(
            "ON: clicking an unlabeled thumbnail runs the latest trained "
            "shipping_label model. Predicted boxes (conf > 0.3) appear as "
            "yellow outlines on the preview. Clicking 'Open in LabelImg' "
            "writes the predictions to the .txt so LabelImg loads them — "
            "you adjust instead of drawing from scratch. "
            "Existing labels are NEVER overwritten silently."
        ),
        disabled=(task != "Shipping Label"),
    )
    if suggestions_on and task == "Shipping Label":
        if not suggestion_weights or not suggestion_weights.exists():
            st.caption(
                f"⚠ Suggestions enabled but no trained model found at "
                f"`{suggestion_weights}`. Train first via the Train tab."
            )
        else:
            st.caption(
                f"💡 Suggestions on — using `{suggestion_weights.relative_to(ROOT)}` "
                f"at conf ≥ {SUGGESTION_CONF_THRESHOLD}"
            )

    # ---- LabelImg launch panel (always visible, top of tab) ----
    with st.container(border=True):
        lc1, lc2 = st.columns([3, 1])
        with lc1:
            if launch_target is not None:
                vid = parse_video_id(launch_target.stem)[0]
                st.markdown(
                    f"**🛠 Label this image**  ·  task: `{task}`  ·  "
                    f"user: `{user}`  ·  "
                    f"opening: `{launch_target.name}` (video `{vid}`, {launch_mode})"
                )
            else:
                st.markdown(
                    f"**🛠 Label this dataset**  ·  task: `{task}`  ·  "
                    f"user: `{user}`  ·  folder: `{folder.relative_to(ROOT)}`"
                )
            if queue:
                labeled_in_q = sum(1 for p in queue if _is_labeled(Path(p)))
                st.caption(
                    f"📋 Queue progress: **{labeled_in_q} / {len(queue)}** labeled  ·  "
                    f"next unlabeled: `{next_in_queue.name if next_in_queue else '(all done)'}`"
                )
        with lc2:
            if launch_target is None:
                st.button(
                    "🚀 Open in LabelImg", use_container_width=True,
                    disabled=True,
                    help="Click a thumbnail or tick the queue checkbox first.",
                    key=f"launch_labelimg_{task}_{version}_{user}_disabled",
                )
            else:
                if st.button(
                    "🚀 Open in LabelImg", type="primary",
                    use_container_width=True,
                    key=f"launch_labelimg_{task}_{version}_{user}",
                ):
                    # Semi-auto labeling: pre-fill <frame>.txt with model
                    # suggestions IF toggle is on AND no existing label.
                    # _prefill returns -1 if existing labels block it (don't
                    # overwrite silently per spec).
                    if (suggestions_on and suggestion_weights
                            and suggestion_weights.exists()):
                        try:
                            preds = _predict_boxes_for_image(
                                str(launch_target),
                                launch_target.stat().st_mtime,
                                str(suggestion_weights),
                                SUGGESTION_CONF_THRESHOLD,
                            )
                        except Exception as e:  # noqa: BLE001
                            preds = []
                            st.warning(f"Suggestion inference failed: {e}")
                        if preds:
                            n = _prefill_label_from_predictions(
                                launch_target, preds, force=False,
                            )
                            if n > 0:
                                st.toast(
                                    f"💡 Pre-filled {n} suggestion(s) into "
                                    f"{launch_target.name}"
                                )
                            elif n == -1:
                                st.toast(
                                    f"💡 Existing label on {launch_target.name} — "
                                    "suggestions NOT applied (no overwrite)"
                                )
                    ok, msg = _launch_labelimg(
                        folder, classes_file, image_path=launch_target,
                    )
                    st.session_state[active_key] = str(launch_target)
                    (st.success if ok else st.error)(msg)
        if launch_target is None:
            st.warning(
                "Please select an image first — click a thumbnail or tick "
                "the **queue** checkbox on one or more frames below."
            )
        rc1, rc2, rc3 = st.columns([4, 2, 2])
        with rc1:
            st.caption(
                "After saving in LabelImg → badges (🟡 → 🟢) update on rerun. "
                "Toggle format to **YOLO** before saving. LabelImg runs detached."
            )
        with rc2:
            auto_refresh = st.checkbox(
                "Auto-refresh (5s)",
                key=f"label_autorefresh_{task}_{version}_{user}",
                help=(
                    "Re-reads the filesystem every 5s — turn ON while labeling, "
                    "OFF when done. Safe to leave on; just causes a periodic redraw."
                ),
            )
        with rc3:
            if st.button("🔄 Refresh now", use_container_width=True,
                         key=f"label_refresh_{task}_{version}_{user}"):
                st.rerun()
        if auto_refresh:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=5000, key=f"label_autorefresh_timer_{task}_{version}_{user}")
        with st.expander("Or copy the command (terminal fallback)"):
            st.code(
                _labelimg_command(folder, classes_file, image_path=active_path),
                language="bash",
            )

    # ---- Per-user stats ----
    my_assigned = [p for p in all_images if assignments.get(p.name) == user]
    my_labeled = sum(1 for p in my_assigned if _is_labeled(p))
    unassigned_total = sum(1 for p in all_images if p.name not in assignments)
    others_assigned = sum(
        1 for p in all_images
        if p.name in assignments and assignments[p.name] != user
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(f"Mine ({user})", f"{my_labeled} / {len(my_assigned)}",
               help="Frames assigned to you that have a saved label.")
    sc2.metric("Unassigned", unassigned_total,
               help="Frames in this task with no owner — anyone can claim.")
    sc3.metric("Others' work", others_assigned,
               help="Frames assigned to other users (read-only for you).")
    sc4.metric("Dataset total", len(all_images))

    # ---- Carry-forward labels from a previous version ----
    other_versions = [v for v in _list_versions(task) if v != version]
    if other_versions:
        with st.expander("📦 Copy labels from a previous version"):
            st.caption(
                "Copies any `.txt` from the source version where the SAME filename "
                "exists in this version AND this version has no `.txt` yet. "
                "**Never overwrites** existing labels here."
            )

            # 1. Source version selector
            source_v = st.selectbox(
                "Source version", other_versions, index=0,
                key=f"carry_source_{task}_{version}_{user}",
            )

            # 2. FPS check between source and destination
            src_dir = _version_dir(task, source_v)
            src_meta = _load_meta(src_dir)
            dst_meta = _load_meta(folder)
            src_fps = src_meta.get("fps")
            dst_fps = dst_meta.get("fps")

            confirm_fps_risk = True  # default: nothing blocking
            if src_fps is not None and dst_fps is not None:
                if abs(float(src_fps) - float(dst_fps)) > 1e-6:
                    st.error(
                        f"⚠ **FPS mismatch detected.** "
                        f"Source `{source_v}` was extracted at **{src_fps} fps**; "
                        f"destination `{version}` at **{dst_fps} fps**. "
                        "Frame `_fNNNN` indexes will not align — carry-forward may "
                        "attach labels to the wrong images."
                    )
                    confirm_fps_risk = st.checkbox(
                        "I understand the risk. Continue anyway.",
                        value=False,
                        key=f"carry_confirm_fps_{task}_{version}_{source_v}_{user}",
                    )
                else:
                    st.caption(f"✓ FPS match: both at **{src_fps} fps**.")
            else:
                st.warning(
                    f"⚠ FPS metadata missing "
                    f"(source: `{src_fps or '—'}`, dest: `{dst_fps or '—'}`). "
                    "Cannot verify frame alignment. Proceed only if you're certain "
                    "the videos were extracted at the same rate. Use **Safe copy** "
                    "below for content-based verification."
                )

            # 3. Safe-mode toggle (hash-based content verification)
            safe_mode = st.checkbox(
                "🛡 Safe copy (verify image content via MD5)",
                value=(src_fps is None or dst_fps is None
                       or (src_fps is not None and dst_fps is not None
                           and abs(float(src_fps) - float(dst_fps)) > 1e-6)),
                key=f"carry_safemode_{task}_{version}_{source_v}_{user}",
                help=(
                    "OFF: copy by filename match (fast). "
                    "ON: hash both files, only copy when image bytes are identical "
                    "(catches mismatches when fps metadata is missing or wrong). "
                    "Hashing is cached — first run is slower, repeats are instant."
                ),
            )

            # 4. Copy button — disabled when fps confirmation is required and missing
            can_copy = bool(confirm_fps_risk)
            if st.button(
                "📦 Copy labels", use_container_width=True,
                disabled=not can_copy,
                key=f"carry_run_{task}_{version}_{source_v}_{user}",
            ):
                copied = 0
                skipped_existing = 0
                no_match = 0
                skipped_hash_mismatch = 0
                for img in all_images:
                    dst_txt = img.with_suffix(".txt")
                    if dst_txt.exists() and dst_txt.stat().st_size > 0:
                        skipped_existing += 1
                        continue
                    src_img = src_dir / img.name
                    src_txt = src_dir / (img.stem + ".txt")
                    if not src_txt.exists() or src_txt.stat().st_size == 0:
                        no_match += 1
                        continue
                    if safe_mode:
                        if not src_img.exists():
                            # No source image to compare against — refuse rather
                            # than silently fall back to filename match.
                            skipped_hash_mismatch += 1
                            continue
                        try:
                            if (_image_md5(str(src_img), src_img.stat().st_mtime)
                                    != _image_md5(str(img), img.stat().st_mtime)):
                                skipped_hash_mismatch += 1
                                continue
                        except OSError:
                            skipped_hash_mismatch += 1
                            continue
                    try:
                        os.link(src_txt, dst_txt)
                    except OSError:
                        shutil.copy2(src_txt, dst_txt)
                    copied += 1

                st.cache_data.clear()
                report = {
                    "copied": copied,
                    "skipped_existing": skipped_existing,
                    "no_match_in_source": no_match,
                }
                if safe_mode:
                    report["skipped_hash_mismatch"] = skipped_hash_mismatch
                st.success(
                    f"📦 Carry-forward from `{source_v}` → `{version}` "
                    f"({'safe' if safe_mode else 'fast'} mode):"
                )
                st.write(report)
                st.rerun()

    # ---- Filters ----
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        status_filter = st.selectbox(
            "Status", ["All", "Unlabeled", "Labeled"], key=status_w,
        )
    with fc2:
        video_ids = sorted({parse_video_id(p.stem)[0] for p in all_images})
        vid_filter = st.selectbox(
            "Video", ["(all)"] + video_ids, key=vid_w,
        )
    with fc3:
        assignment_filter = st.selectbox(
            "Assignment",
            ["My work", "Mine only", "Unassigned (claim)", "All"],
            key=asg_w,
            help=(
                "My work: yours + unassigned (default).  "
                "Mine only: just yours.  "
                "Unassigned: claim from the pool.  "
                "All: everyone's (read-only for others')."
            ),
        )

    # Apply assignment filter FIRST so subsequent filters operate on the
    # subset visible to this user.
    if assignment_filter == "My work":
        # Default view: assigned-to-me OR unassigned (the work pool).
        filtered = [
            p for p in all_images
            if assignments.get(p.name) in (user, None)
        ]
    elif assignment_filter == "Mine only":
        filtered = [p for p in all_images if assignments.get(p.name) == user]
    elif assignment_filter == "Unassigned (claim)":
        filtered = [p for p in all_images if p.name not in assignments]
    else:  # All
        filtered = list(all_images)

    if status_filter == "Unlabeled":
        filtered = [p for p in filtered if not _is_labeled(p)]
    elif status_filter == "Labeled":
        filtered = [p for p in filtered if _is_labeled(p)]
    if vid_filter != "(all)":
        filtered = [p for p in filtered if parse_video_id(p.stem)[0] == vid_filter]

    # ---- Per-user progress bar ----
    if my_assigned:
        st.progress(
            (my_labeled / len(my_assigned)),
            text=f"{user}: {my_labeled} / {len(my_assigned)} of your frames labeled",
        )
    else:
        st.progress(
            0.0, text=f"{user}: 0 frames assigned yet — claim some below 👇",
        )

    if not filtered:
        wc1, wc2 = st.columns([3, 1])
        with wc1:
            st.warning(
                "No images match the current filters. "
                "Try changing Status / Video / Assignment, or click **Reset filters** →"
            )
        with wc2:
            if st.button("🔄 Reset filters", use_container_width=True,
                         key=f"reset_filters_{task}_{version}_{user}"):
                st.session_state[f"_pending_filter_reset::{task}::{version}::{user}"] = True
                st.rerun()
        return

    # ---- Active-image preview (read-only) — active_path was resolved above ----
    if active_path is not None:
        # Snapshot taken when user clicked a thumbnail (see grid below).
        # Locks prev/next navigation to the filter context at click time.
        snap_paths = st.session_state.get(snap_key) or []
        nav_list = [Path(s) for s in snap_paths if Path(s).exists()]
        if active_path not in nav_list:
            nav_list = [active_path] + nav_list
        _preview_panel(
            task, active_path, nav_list, class_names,
            suggestion_weights=suggestion_weights,
            suggestions_on=suggestions_on,
        )
        st.markdown("---")

    # ---- Pagination ----
    total = len(filtered)
    page_count = max(1, (total + GRID_PAGE_SIZE - 1) // GRID_PAGE_SIZE)
    page = st.number_input(
        f"Page (1–{page_count})",
        min_value=1, max_value=page_count, value=1, step=1,
        key=page_w,
    )
    page_slice = filtered[(page - 1) * GRID_PAGE_SIZE : page * GRID_PAGE_SIZE]
    st.write(
        f"**{total}** match  ·  page **{page}/{page_count}** "
        f"({len(page_slice)} on this page)  ·  queue **{len(queue)}**"
    )

    # ---- Bulk queue + claim actions (above grid so widget writes happen
    # before checkboxes are instantiated below). ----
    qb1, qb2, qb3 = st.columns(3)
    with qb1:
        if st.button("📋 Queue all unlabeled on this page",
                     use_container_width=True,
                     key=f"queue_addpage_{task}_{version}_{user}_{page}"):
            for p in page_slice:
                if _is_labeled(p):
                    continue
                k = str(p)
                if k not in queue:
                    queue.append(k)
                st.session_state[f"{queue_chk_prefix}{p.name}"] = True
            st.rerun()
    with qb2:
        if st.button("🗑 Clear queue", use_container_width=True,
                     key=f"queue_clear_{task}_{version}_{user}",
                     disabled=not queue):
            for p_str in queue:
                ck = f"{queue_chk_prefix}{Path(p_str).name}"
                if ck in st.session_state:
                    st.session_state[ck] = False
            queue.clear()
            st.rerun()
    with qb3:
        # "Claim queued" — assigns every queued frame that's currently
        # unassigned to the current user. Frames already assigned to others
        # are left alone (no stealing).
        claimable = [
            Path(p) for p in queue
            if Path(p).exists() and Path(p).name not in assignments
        ]
        if st.button(
            f"👤 Claim {len(claimable)} unassigned in queue",
            use_container_width=True, disabled=not claimable,
            key=f"queue_claim_{task}_{version}_{user}",
            help="Assigns every queued frame that has no owner to you. "
                 "Frames already assigned to other users are not touched.",
        ):
            updated = dict(assignments)
            for p in claimable:
                updated[p.name] = user
            _save_assignments(task, version, updated)
            st.toast(f"Claimed {len(claimable)} frame(s) for {user}.")
            st.rerun()

    # ---- Bulk exclude / include actions on the queue ----
    excl_chk_prefix = f"exclude_chk_{task}_{version}_{user}_"
    eb1, eb2 = st.columns(2)
    with eb1:
        queued_existing = [Path(p) for p in queue if Path(p).exists()]
        if st.button(
            f"🚫 Exclude {len(queued_existing)} queued from training",
            use_container_width=True, disabled=not queued_existing,
            key=f"queue_exclude_{task}_{version}_{user}",
            help="Marks queued frames as excluded — split.py and Clean dataset "
                 "will skip them. Doesn't delete files.",
        ):
            new_excluded = set(excluded) | {p.name for p in queued_existing}
            _save_excluded(task, version, new_excluded)
            for p in queued_existing:
                st.session_state[f"{excl_chk_prefix}{p.name}"] = True
            st.toast(f"Excluded {len(queued_existing)} frame(s) from training.")
            st.rerun()
    with eb2:
        if st.button(
            "♻️ Re-include all excluded",
            use_container_width=True, disabled=not excluded,
            key=f"queue_include_{task}_{version}_{user}",
            help="Clears the exclude list for this version (does not delete files).",
        ):
            for name in list(excluded):
                st.session_state[f"{excl_chk_prefix}{name}"] = False
            _save_excluded(task, version, set())
            st.toast("All exclusions cleared.")
            st.rerun()

    # ---- Grid ----
    cols_per_row = 4
    for row_start in range(0, len(page_slice), cols_per_row):
        row = page_slice[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, row):
            with col:
                rgb = _load_thumbnail(str(p), p.stat().st_mtime, GRID_THUMB_W,
                                      square=True)
                st.image(rgb, use_container_width=True)
                # Status badge
                badge = "🟢" if _is_labeled(p) else "🟡"
                # Assignment badge — quick visual of who owns this frame
                owner = assignments.get(p.name)
                if owner == user:
                    asg = "👤"           # mine
                elif owner is None:
                    asg = "🆓"           # unassigned (claimable)
                else:
                    asg = "👥"           # someone else's (read-only-ish)
                short = p.stem.split("_f")[-1] if "_f" in p.stem else p.stem
                excl_flag = "🚫" if p.name in excluded else ""
                is_active = active_path is not None and Path(p) == active_path
                if st.button(
                    f"{badge}{asg}{excl_flag} {short}",
                    key=f"open_{task}_{version}_{user}_{p.name}",
                    use_container_width=True,
                    type=("primary" if is_active else "secondary"),
                ):
                    st.session_state[f"_pending_active::{task}::{version}::{user}"] = str(p)
                    st.session_state[snap_key] = [str(x) for x in filtered]
                    st.rerun()
                # Queue + Use-for-training toggles on a single horizontal row
                # so card heights stay aligned across the grid.
                p_str = str(p)
                in_queue = p_str in queue
                qcol, ucol = st.columns(2)
                with qcol:
                    checked = st.checkbox(
                        "📋 Queue", value=in_queue,
                        key=f"{queue_chk_prefix}{p.name}",
                    )
                with ucol:
                    use_for_training = st.checkbox(
                        "✅ Use", value=p.name not in excluded,
                        key=f"{excl_chk_prefix}{p.name}",
                        help="Use this frame for training (inverse of exclude). "
                             "Unchecking adds it to _excluded.json.",
                    )
                if checked and p_str not in queue:
                    queue.append(p_str)
                elif not checked and p_str in queue:
                    queue.remove(p_str)
                if use_for_training and p.name in excluded:
                    excluded.discard(p.name)
                    _save_excluded(task, version, excluded)
                elif not use_for_training and p.name not in excluded:
                    excluded.add(p.name)
                    _save_excluded(task, version, excluded)
                # OCR text — show user correction if any, else auto. Icon
                # tells the reviewer at a glance whether it's been verified.
                ocr_boxes = _load_ocr(p)
                if ocr_boxes:
                    first_text, first_corrected = _resolved_ocr_text(ocr_boxes[0])
                    if first_text:
                        icon = "✏" if first_corrected else "💬"
                        snippet = first_text[:24] + ("…" if len(first_text) > 24 else "")
                        st.caption(f"{icon} `{snippet}`")

    # ---- Debug expander — moved to bottom; primary state is now visible
    # via the metric row at top of the page. ----
    with st.expander("🔧 Debug: selection + assignment state"):
        st.write({
            "task": task, "user": user,
            "active_image_path": str(active_path) if active_path else None,
            "active_video_id": parse_video_id(active_path.stem)[0] if active_path else None,
            "video_filter": current_vid,
            "previous_video_filter": prev_vid,
            "assignment_filter": assignment_filter,
            "status_filter": status_filter,
            "queue_count": len(queue),
            "queue_labeled_count": sum(1 for p in queue if _is_labeled(Path(p))),
            "next_in_queue": str(next_in_queue) if next_in_queue else None,
            "launch_target": str(launch_target) if launch_target else None,
            "launch_mode": launch_mode,
            "assignments_total": len(assignments),
            "my_assigned_count": len(my_assigned),
            "my_labeled_count": my_labeled,
            "assignments_path": str(_assignments_path(task, version)),
        })


# ---------- Train tab ----------

LOGS_DIR = ROOT / "logs"
TRAIN_STALE_SECONDS = 300  # if log mtime older than this AND state=running → assume crashed


def _train_state_path(task: str, version: str) -> Path:
    return LOGS_DIR / f"train_{_task_slug(task)}_{version}.state.json"


def _train_log_path(task: str, version: str) -> Path:
    return LOGS_DIR / f"train_{_task_slug(task)}_{version}.log"


def _read_train_state(task: str, version: str) -> dict:
    """Returns the training state dict, applying a stale-run fixup."""
    p = _train_state_path(task, version)
    if not p.exists():
        return {"status": "idle"}
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle"}
    # Stale detection: state says "running" but log file hasn't been touched
    # in TRAIN_STALE_SECONDS → wrapper crashed without finalizing.
    if state.get("status") == "running":
        log_p = _train_log_path(task, version)
        if log_p.exists():
            age = time.time() - log_p.stat().st_mtime
            if age > TRAIN_STALE_SECONDS:
                state["status"] = "failed"
                state["stale"] = True
                state["stale_log_age_sec"] = round(age)
        else:
            # No log file yet — only stale if state has been around > 30s
            age = time.time() - state.get("started_at", time.time())
            if age > 30:
                state["status"] = "failed"
                state["stale"] = True
    return state


def _spawn_training(task: str, version: str, data_yaml: Path, name: str,
                    epochs: int, batch: int, imgsz: int, device: str) -> tuple[bool, str]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "scripts" / "_train_runner.py"
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists() or not runner.exists():
        return False, f"Missing executable: venv_py={venv_py}, runner={runner}"

    log_path = _train_log_path(task, version)
    state_path = _train_state_path(task, version)
    cmd = [
        str(venv_py), str(runner),
        "--data", str(data_yaml),
        "--name", name,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", device,
        "--state-file", str(state_path),
    ]
    try:
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=flags, close_fds=True, cwd=str(ROOT),
        )
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, f"Spawned training → log: {log_path.relative_to(ROOT)}"


def _tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        # Cheap tail — fine for our log sizes (kilobytes per minute).
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"(could not read log: {e})"
    return "\n".join(lines[-n:]) or "(empty log)"


def _count_split(data_root: Path, split: str) -> tuple[int, int]:
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    n_img = sum(1 for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
            if img_dir.exists() else 0
    n_lbl = sum(1 for p in lbl_dir.iterdir()
                if p.is_file() and p.suffix == ".txt") if lbl_dir.exists() else 0
    return n_img, n_lbl


def _resolve_data_root_for_task(task: str) -> Path:
    """Mirror split.py's resolution: read data.yaml.path, resolve, fallback."""
    data_yaml = TASK_CONFIG[task]["data_yaml"]
    if not data_yaml.exists():
        return TASK_CONFIG[task]["image_dir"]
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        path_field = cfg.get("path", ".")
        for cand in ((ROOT / path_field), (data_yaml.parent / path_field)):
            r = cand.resolve()
            if r.exists():
                return r
    except Exception:
        pass
    return TASK_CONFIG[task]["image_dir"]


def _ensure_version_data_yaml(task: str, version: str) -> Path:
    """Generate (or refresh) a per-version data.yaml inside the version folder.

    Layout it expects after split.py runs:
        <version_dir>/
            <video_id>_fNNNN.jpg     (source frames, root-level)
            <video_id>_fNNNN.txt     (LabelImg labels, root-level)
            images/{train,val}/      (split.py output)
            labels/{train,val}/      (split.py output)
            _data.yaml               (this file)

    `path:` is absolute to dodge YOLO's relative-path resolution quirks.
    """
    version_dir = _version_dir(task, version)
    version_dir.mkdir(parents=True, exist_ok=True)

    class_names, _ = load_class_names(version_dir)
    if not class_names:
        # Final fallback — read the task's predefined_classes.txt directly.
        cfile = TASK_CONFIG[task]["classes_file"]
        if cfile.exists():
            lines = [ln.strip() for ln in cfile.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            class_names = {i: n for i, n in enumerate(lines)}

    parts = [
        f"# Auto-generated for task={task} version={version}",
        f"path: {version_dir.resolve().as_posix()}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"names:",
    ]
    for cid, cname in sorted(class_names.items()):
        parts.append(f"  {cid}: {cname}")
    parts.append("")  # trailing newline

    yaml_path = version_dir / "_data.yaml"
    yaml_path.write_text("\n".join(parts), encoding="utf-8")
    return yaml_path


def train_tab(task: str, version: str) -> None:
    st.title(f"🚀 Train — {task} / {version}")

    cfg = TASK_CONFIG[task]
    # Per-version data.yaml is the source of truth for both split + train.
    data_yaml = _ensure_version_data_yaml(task, version)
    # The Train tab now reads metrics from the VERSION folder, not the global
    # task data root. Each version trains on its own images/{train,val}.
    data_root = _version_dir(task, version)

    state = _read_train_state(task, version)
    status = state.get("status", "idle")

    # ---- Status banner ----
    if status == "running":
        step = state.get("step", "?")
        started = state.get("started_at", 0)
        elapsed = int(time.time() - started) if started else 0
        st.info(f"⏳ Training in progress — step `{step}` · {elapsed}s elapsed")
    elif status == "done":
        ended = state.get("ended_at", 0)
        st.success(
            f"✅ Last run: success · "
            f"finished {datetime.fromtimestamp(ended).strftime('%Y-%m-%d %H:%M:%S') if ended else '?'}"
        )
    elif status == "failed":
        msg = "❌ Last run: failed"
        if state.get("stale"):
            msg += f" (stale — log untouched for >{state.get('stale_log_age_sec', '?')}s; "
            msg += "wrapper likely crashed)"
        elif state.get("exit_code") is not None:
            msg += f" (exit code {state['exit_code']}, step `{state.get('step', '?')}`)"
        st.error(msg)
    else:
        st.caption(f"Status: **idle** — no run yet for `{task}` / `{version}`.")

    # ---- Dataset status ----
    src_imgs = sum(1 for _ in data_root.glob("*.jpg")) + sum(1 for _ in data_root.glob("*.png"))
    src_lbls = sum(1 for p in data_root.glob("*.txt")
                   if p.name not in ("classes.txt", "predefined_classes.txt")
                   and p.stat().st_size > 0)
    excluded = _load_excluded(task, version)
    excluded_count = len(excluded)

    MIN_FRAMES = 40
    ready = src_lbls >= MIN_FRAMES
    progress_val = min(src_lbls / MIN_FRAMES, 1.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng ảnh", src_imgs)
    c2.metric("Đã có nhãn", src_lbls)
    c3.metric("Bỏ qua", excluded_count, help="Frames bị loại khỏi training")

    if ready:
        st.progress(1.0, text=f"✅ Đủ {src_lbls} frames có nhãn — sẵn sàng train!")
    else:
        st.progress(progress_val,
                    text=f"⏳ {src_lbls}/{MIN_FRAMES} frames có nhãn — cần thêm {MIN_FRAMES - src_lbls} nữa")

    train_n_img, train_n_lbl = _count_split(data_root, "train")
    val_n_img, val_n_lbl = _count_split(data_root, "val")
    split_total_lbls = train_n_lbl + val_n_lbl
    img_train_dir = data_root / "images" / "train"
    last_split_at = (
        datetime.fromtimestamp(img_train_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        if img_train_dir.exists() else "chưa có"
    )
    drift = src_lbls - split_total_lbls
    if drift > 0:
        st.warning(f"⚠ Có **{drift}** nhãn mới chưa được đưa vào split — bấm **Retrain** để cập nhật.")

    with st.expander(f"Chi tiết split (lần cuối: {last_split_at})"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Train ảnh", train_n_img)
        sc2.metric("Train nhãn", train_n_lbl)
        sc3.metric("Val ảnh", val_n_img)
        sc4.metric("Val nhãn", val_n_lbl)

    # ---- Config (run name + hyperparams) ----
    with st.expander("⚙️ Cấu hình training"):
        name_default = f"rl_{_task_slug(task)}_{version}"
        name = st.text_input(
            "Run name", value=name_default, key=f"train_name_{task}_{version}",
            help="Output: models/<name>/weights/best.pt. Bump suffix (-v2, -v3…) to keep old weights.",
        )
        best_pt = ROOT / "models" / name / "weights" / "best.pt"
        if best_pt.exists():
            ts = datetime.fromtimestamp(best_pt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            st.success(f"📦 `{best_pt.relative_to(ROOT)}` · {ts} · {size_mb:.1f} MB")
        else:
            st.caption(f"`models/{name}/weights/best.pt` — chưa có.")
        hc1, hc2, hc3, hc4 = st.columns(4)
        epochs = hc1.number_input("Epochs", min_value=1, max_value=500, value=30, step=10,
                                  key=f"train_epochs_{task}_{version}")
        batch = hc2.number_input("Batch", min_value=1, max_value=64, value=4,
                                 key=f"train_batch_{task}_{version}")
        imgsz = hc3.number_input("Img size", min_value=320, max_value=1280, value=640, step=64,
                                 key=f"train_imgsz_{task}_{version}")
        device = hc4.selectbox("Device", ["cpu", "0"], index=0,
                               key=f"train_device_{task}_{version}",
                               help="cpu = CPU; 0 = GPU 0")
    # name/epochs/batch/imgsz/device default if expander not opened
    name = st.session_state.get(f"train_name_{task}_{version}", f"rl_{_task_slug(task)}_{version}")
    epochs = st.session_state.get(f"train_epochs_{task}_{version}", 30)
    batch = st.session_state.get(f"train_batch_{task}_{version}", 4)
    imgsz = st.session_state.get(f"train_imgsz_{task}_{version}", 640)
    device = st.session_state.get(f"train_device_{task}_{version}", "cpu")
    best_pt = ROOT / "models" / name / "weights" / "best.pt"

    # ---- Guards ----
    is_running = status == "running"
    LABEL_THRESHOLD = 10
    EXCLUDED_RATIO_WARN = 0.5
    excluded_ratio = (excluded_count / src_imgs) if src_imgs else 0.0
    guard_msgs: list[str] = []
    if src_lbls < LABEL_THRESHOLD:
        guard_msgs.append(f"Chỉ có **{src_lbls}** frame có nhãn — nên có ≥{LABEL_THRESHOLD}.")
    if excluded_ratio > EXCLUDED_RATIO_WARN:
        guard_msgs.append(
            f"**{excluded_count}/{src_imgs}** frames bị loại ({excluded_ratio:.0%}) — quá nhiều.")
    if guard_msgs:
        st.warning("⚠ Dataset chưa đủ:\n\n" + "\n".join(f"- {m}" for m in guard_msgs))

    can_train = train_n_img > 0 and val_n_img > 0 and train_n_lbl > 0
    if not can_train:
        st.warning(
            f"`{version}` chưa sẵn sàng train — cần ít nhất 1 ảnh train, 1 ảnh val, 1 nhãn train. "
            "Bấm **Retrain** để split trước."
        )

    confirm = st.checkbox(
        f"Xác nhận train **{task}** / `{version}` — {epochs} epoch, device `{device}`",
        key=f"train_confirm_{task}_{version}", value=False,
    )
    rc1, rc2, rc3 = st.columns([3, 2, 1])
    with rc1:
        if st.button(
            f"🚀 Retrain",
            type="primary", use_container_width=True,
            disabled=is_running or not confirm,
            key=f"train_go_{task}_{version}",
        ):
            data_yaml = _ensure_version_data_yaml(task, version)
            ok, msg = _spawn_training(
                task=task, version=version, data_yaml=data_yaml, name=name,
                epochs=int(epochs), batch=int(batch),
                imgsz=int(imgsz), device=device,
            )
            (st.success if ok else st.error)(msg)
            time.sleep(0.5)
            st.rerun()
    with rc2:
        if st.button("🔁 Re-split",
                     use_container_width=True,
                     disabled=is_running,
                     key=f"train_resplit_{task}_{version}",
                     help="Chạy split.py, cập nhật train/val metrics mà không train lại."):
            data_yaml = _ensure_version_data_yaml(task, version)
            venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
            try:
                r = subprocess.run(
                    [str(venv_py), str(ROOT / "scripts" / "split.py"),
                     "--data", str(data_yaml)],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=60,
                )
                if r.returncode == 0:
                    st.success("✅ Split xong — metrics đã cập nhật.")
                else:
                    st.error(f"split.py lỗi (exit {r.returncode}):\n{r.stderr or r.stdout}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Re-split thất bại: {type(e).__name__}: {e}")
            st.cache_data.clear()
            st.rerun()
    with rc3:
        if st.button("🔄", use_container_width=True,
                     key=f"train_refresh_{task}_{version}",
                     help="Refresh log + trạng thái."):
            st.rerun()

    # ---- Tail of logs ----
    log_path = _train_log_path(task, version)
    st.markdown(
        f"**Log** · "
        f"`{log_path.relative_to(ROOT) if log_path.exists() else '(chưa có log)'}`"
    )
    st.code(_tail_lines(log_path, 20), language="text")

    # ---- Dataset management (clean + archive) — ít dùng, để cuối ----
    with st.expander("🗂 Quản lý dataset"):
        dm1, dm2 = st.columns(2)
        with dm1:
            st.markdown("**🧹 Clean dataset**")
            require_correct = st.checkbox(
                "Chỉ lấy frame được Review đánh dấu `is_correct=yes`",
                value=False, key=f"clean_require_correct_{task}_{version}",
            )
            confirm_clean = st.checkbox(
                "Tôi hiểu: thao tác này tạo version MỚI",
                value=False, key=f"clean_confirm_{task}_{version}",
            )
            if st.button(
                "🧹 Clean → tạo version mới",
                use_container_width=True,
                disabled=not confirm_clean or src_lbls == 0,
                key=f"clean_run_{task}_{version}",
            ):
                with st.spinner("Cleaning…"):
                    new_v, stats = _clean_version(
                        task, version, require_review_correct=require_correct,
                    )
                st.session_state[f"_pending_version_switch::{task}"] = new_v
                st.cache_data.clear()
                st.success(
                    f"✅ Tạo `{new_v}` với **{stats['copied']}** frame. "
                    f"Bỏ qua: chưa nhãn={stats['skipped_unlabeled']}, "
                    f"excluded={stats['skipped_excluded']}, "
                    f"review-rejected={stats['skipped_review_no']}."
                )
                time.sleep(0.5)
                st.rerun()

        with dm2:
            st.markdown("**📦 Archive version**")
            if version == LEGACY_VERSION:
                st.caption("`v_legacy` không thể archive.")
            else:
                is_arch = _is_archived(task, version)
                if is_arch:
                    st.caption(f"`{version}` đang **archived**.")
                    if st.button("♻️ Un-archive", use_container_width=True,
                                 key=f"unarchive_{task}_{version}"):
                        _set_archived(task, version, False)
                        st.toast(f"`{version}` đã un-archive.")
                        st.rerun()
                else:
                    confirm_arch = st.checkbox(
                        "Xác nhận archive (ẩn version, giữ nguyên file)",
                        value=False, key=f"arch_confirm_{task}_{version}",
                    )
                    if st.button(
                        f"📦 Archive `{version}`",
                        use_container_width=True,
                        disabled=not confirm_arch,
                        key=f"archive_{task}_{version}",
                    ):
                        _set_archived(task, version, True)
                        st.toast(f"`{version}` đã archive.")
                        st.rerun()

    # ---- Auto-refresh while running ----
    if is_running:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, key=f"train_auto_{task}_{version}")


def _login_gate() -> bool:
    """Return True if the user is authenticated. Show login form otherwise."""
    if st.session_state.get("authenticated"):
        return True

    try:
        app_password = st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        # No password configured — allow access (local dev or unprotected deploy).
        return True

    st.title("🔐 Inspection App")
    st.markdown("Nhập mật khẩu để tiếp tục.")
    pwd = st.text_input("Mật khẩu", type="password", key="_login_pwd")
    if st.button("Đăng nhập", type="primary"):
        if pwd == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Sai mật khẩu.")
    return False


def main() -> None:
    st.set_page_config(page_title="Inspection App", page_icon="📦", layout="wide")
    if not _login_gate():
        st.stop()
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Styrene+B:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

        /* === BASE — Claude.ai palette === */
        /* bg: #F5F0E8  sidebar: #1C1917  accent: #D97706 (amber) */
        html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
        .stApp { background-color: #F5F0E8; }
        .block-container { padding-top: 1.75rem !important; padding-bottom: 2.5rem !important; }

        /* === SIDEBAR — Claude dark === */
        [data-testid="stSidebar"] {
            background-color: #1C1917;
            border-right: 1px solid #292524;
        }
        [data-testid="stSidebar"] * { color: #A8A29E; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stCheckbox span p,
        [data-testid="stSidebar"] small {
            color: #78716C !important;
            font-size: 0.68rem !important; font-weight: 700 !important;
            text-transform: uppercase; letter-spacing: 0.09em;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #292524 !important; border-color: #3C3733 !important;
            border-radius: 10px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div { color: #E7E5E4 !important; }
        [data-testid="stSidebar"] svg { fill: #57534E !important; }
        [data-testid="stSidebar"] hr { border-color: #292524 !important; margin: 0.6rem 0 !important; }
        [data-testid="stSidebar"] button {
            background: #292524 !important; color: #A8A29E !important;
            border: 1px solid #3C3733 !important; border-radius: 8px !important;
            font-size: 0.78rem !important; font-weight: 600 !important;
        }
        [data-testid="stSidebar"] button:hover {
            background: #3C3733 !important; color: #F59E0B !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] > div {
            background: #292524 !important; border-color: #3C3733 !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] [role="button"] {
            color: #A8A29E !important; background: #292524 !important;
        }

        /* === TABS — pill như Claude === */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: #E8E0D6;
            border-radius: 12px; padding: 4px; gap: 2px;
            border-bottom: none !important; width: fit-content;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            background: transparent; border-radius: 9px; border: none;
            padding: 0.45rem 1rem; font-size: 0.82rem; font-weight: 600;
            color: #78716C; transition: all 0.15s ease;
        }
        [data-testid="stTabs"] [data-baseweb="tab"]:hover { color: #B45309; background: #F5EFE6; }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: #FFFFFF !important; color: #B45309 !important;
            box-shadow: 0 1px 5px rgba(0,0,0,0.12) !important;
        }
        [data-testid="stTabs"] [data-testid="stMarkdownContainer"] { padding-top: 1.5rem; }

        /* === HEADINGS === */
        h1 { font-size: 1.5rem !important; font-weight: 700 !important;
             color: #1C1917 !important; letter-spacing: -0.02em; margin-bottom: 0.1rem !important; }
        h2 { font-size: 1rem !important; font-weight: 700 !important; color: #292524 !important; }
        h3 { font-size: 0.72rem !important; font-weight: 700 !important; color: #A8A29E !important;
             text-transform: uppercase; letter-spacing: 0.1em; }

        /* === CARDS (metric, expander, container) === */
        [data-testid="stMetric"],
        div[data-testid="metric-container"] {
            background: #FFFBF5 !important;
            border: 1px solid #D6CEC4 !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            box-shadow: 0 1px 4px rgba(28,25,23,0.06) !important;
        }
        [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
            color: #A8A29E !important; font-size: 0.67rem !important;
            font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.09em !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
            color: #1C1917 !important; font-weight: 700 !important;
        }

        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        div[data-testid="stExpander"] > div {
            background: #FFFBF5 !important;
            border: 1px solid #D6CEC4 !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 3px rgba(28,25,23,0.05) !important;
            margin-bottom: 0.5rem; overflow: hidden;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] [role="button"] {
            font-weight: 600 !important; color: #292524 !important;
            padding: 0.85rem 1.1rem !important; font-size: 0.85rem !important;
            background: #FFFBF5 !important;
        }
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] [role="button"]:hover { background: #F5EDE0 !important; }

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #FFFBF5 !important;
            border: 1px solid #D6CEC4 !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 4px rgba(28,25,23,0.06) !important;
            padding: 1.1rem !important;
        }

        /* === BUTTONS === */
        button[kind="primary"] {
            background: #D97706 !important; border: none !important;
            border-radius: 10px !important; font-weight: 600 !important;
            color: #FFFFFF !important;
            box-shadow: 0 1px 6px rgba(217,119,6,0.35) !important;
            transition: all 0.16s ease !important;
        }
        button[kind="primary"]:hover:not(:disabled) {
            background: #B45309 !important;
            box-shadow: 0 3px 14px rgba(217,119,6,0.45) !important;
            transform: translateY(-1px) !important;
        }
        button[kind="secondary"] {
            border-radius: 10px !important; border: 1px solid #D6CEC4 !important;
            font-weight: 600 !important; color: #57534E !important;
            background: #FFFBF5 !important; transition: all 0.15s ease !important;
        }
        button[kind="secondary"]:hover:not(:disabled) {
            border-color: #F59E0B !important; color: #B45309 !important;
            background: #FFFBEE !important;
        }
        button:disabled { opacity: 0.38 !important; }

        /* === INPUTS === */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input, textarea {
            border-radius: 10px !important; border-color: #D6CEC4 !important;
            background: #FFFBF5 !important; color: #1C1917 !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus, textarea:focus {
            border-color: #D97706 !important;
            box-shadow: 0 0 0 3px rgba(217,119,6,0.14) !important;
        }
        [data-baseweb="select"] > div {
            border-radius: 10px !important; border-color: #D6CEC4 !important;
            background: #FFFBF5 !important;
        }

        /* === FILE UPLOADER === */
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            background: #FFFBF5 !important;
            border: 2px dashed #C4BAB0 !important;
            border-radius: 16px !important; transition: border-color 0.2s;
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: #D97706 !important; }

        /* === ALERTS === */
        [data-testid="stAlert"] {
            border-radius: 12px !important; border-left-width: 4px !important;
            font-size: 0.84rem !important; background: #FFFBF5 !important;
        }

        /* === PROGRESS BAR === */
        [data-testid="stProgressBar"] > div { border-radius: 999px; background: #E8D9B8; }
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #B45309, #D97706) !important;
            border-radius: 999px;
        }

        /* === IMAGES === */
        div[data-testid='stImage'] img {
            max-height: 60vh; width: auto !important; margin: 0 auto; display: block;
            border-radius: 12px; box-shadow: 0 4px 18px rgba(28,25,23,0.12);
            transition: transform 0.12s ease;
        }
        div[data-testid='stImage'] img:hover { transform: scale(1.03); }

        /* === CODE === */
        [data-testid="stCode"] { border-radius: 10px !important; font-size: 0.78rem !important; }

        /* === MISC === */
        hr { border-color: #D6CEC4 !important; margin: 1.25rem 0 !important; }
        [data-testid="stCaptionContainer"] p { color: #A8A29E !important; font-size: 0.8rem !important; }
        [data-testid="stCheckbox"] span { color: #57534E !important; font-size: 0.85rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    st.sidebar.markdown(
        """
        <div style="padding:1.4rem 1rem 1rem; display:flex; align-items:center; gap:0.65rem;">
            <div style="background:#D97706; border-radius:10px; width:34px; height:34px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.05rem; flex-shrink:0; box-shadow:0 2px 8px rgba(217,119,6,0.4);">📦</div>
            <div>
                <div style="font-size:0.85rem; font-weight:700; color:#E7E5E4;
                            letter-spacing:-0.01em; line-height:1.2;">Inspection</div>
                <div style="font-size:0.67rem; color:#57534E; font-weight:500;
                            text-transform:uppercase; letter-spacing:0.07em;">Return QC · AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # One shared task selector — both tabs read this. Streamlit tabs do not
    # scope sidebar widgets, so a per-tab selectbox would duplicate.
    task = st.sidebar.selectbox("Task", list(TASK_CONFIG.keys()), index=0)

    # Apply pending version switch BEFORE the version selectbox is created.
    # Set by import_tab's Send button when a new version is born — see there.
    pending_v = st.session_state.pop(f"_pending_version_switch::{task}", None)
    if pending_v:
        st.session_state[f"current_version::{task}"] = pending_v

    show_archived = st.session_state.get(f"show_archived::{task}", False)
    versions = _list_versions(task, include_archived=show_archived)
    if not versions:
        versions = [LEGACY_VERSION]
    version = st.sidebar.selectbox(
        "Version", versions, index=0,
        key=f"current_version::{task}",
        help="Newest version is on top. v_legacy = pre-versioned data.",
    )
    st.sidebar.markdown("---")

    # User identity + archived toggle
    with st.sidebar.expander("⚙️ Cài đặt"):
        st.selectbox("Người dùng", USERS, index=0, key="current_user")
        st.checkbox(
            "Hiện archived versions", value=False, key=f"show_archived_adv::{task}",
        )
    if st.session_state.get(f"show_archived_adv::{task}") != st.session_state.get(f"show_archived::{task}"):
        st.session_state[f"show_archived::{task}"] = st.session_state.get(f"show_archived_adv::{task}", False)

    # Cloud sync (only shown when HF token is configured)
    if os.environ.get("HUGGING_FACE_ACCESS_TOKEN"):
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "<small>☁️ &nbsp;CLOUD SYNC</small>", unsafe_allow_html=True
        )
        col1, col2 = st.sidebar.columns(2)
        if col1.button("↓ Pull", help="Download latest from HF Hub"):
            with st.spinner("Pulling…"):
                try:
                    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from hf_sync import pull_from_hub
                    pull_from_hub(ROOT)
                    st.sidebar.success("Pulled ✓")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(str(e))
        if col2.button("↑ Push", help="Upload local changes to HF Hub"):
            with st.spinner("Pushing…"):
                try:
                    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from hf_sync import push_to_hub
                    push_to_hub(ROOT)
                    st.sidebar.success("Pushed ✓")
                except Exception as e:
                    st.sidebar.error(str(e))

    _ultralytics_available = True
    try:
        import ultralytics  # noqa: F401
        if ultralytics.YOLO is None:
            _ultralytics_available = False
    except ImportError:
        _ultralytics_available = False

    if _ultralytics_available:
        tab_import, tab_label, tab_review, tab_train, tab_pipeline = st.tabs(
            ["⬆ Import", "🏷 Label", "🔍 Review", "🚀 Train", "▶ Pipeline"]
        )
        with tab_train:
            train_tab(task, version)
        with tab_pipeline:
            pipeline_tab()
    else:
        tab_import, tab_label, tab_review = st.tabs(
            ["⬆ Import", "🏷 Label", "🔍 Review"]
        )

    with tab_import:
        _empty_state = not _list_versions(task)
        if _empty_state:
            st.info("👋 Bắt đầu ở đây: import video hoặc ảnh vào dataset.")
        import_tab(task, version)
    with tab_label:
        label_tab(task, version)
    with tab_review:
        review_tab(task, version)


if __name__ == "__main__":
    main()
