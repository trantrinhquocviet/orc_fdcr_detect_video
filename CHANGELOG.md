# Changelog

Track model versions and pipeline changes. Follow the format below for each
new training run or significant pipeline change.

## Template

```
## [vN] YYYY-MM-DD — short title

### Model
- Run name: rl_yolov8n_vN
- Base weights: yolov8n.pt
- Train config: epochs=N, imgsz=640, batch=N, device=cpu
- Dataset: <X> labeled frames across <Y> videos (<train_videos> train / <val_videos> val)
- Class balance: damaged_item=<N>, empty_box=<N>

### Metrics (val)
- mAP@50: ___
- mAP@50-95: ___
- damaged_item P/R: ___ / ___
- empty_box   P/R: ___ / ___

### Pipeline / script changes
- (bullet point each)

### Known issues / what to fix next
- (bullet point each)
```

---

## [v1] 2026-04-30 — first end-to-end pipeline smoke test

### Model
- Run name: `rl_yolov8n_v1`
- Base weights: `yolov8n.pt`
- Train config: `epochs=10, imgsz=640, batch=4, device=cpu`
- Dataset: 5 labeled frames in 1 video (`TTKMFB581`); fallback per-image split (now removed)
- Class balance: `damaged_item=5`, `empty_box=0`

### Metrics (val on 1 frame)
- mAP@50: 0.995 (statistical noise — 1 val instance)
- mAP@50-95: 0.42
- damaged_item P/R: 0.0025 / 1.0 (predicting damage everywhere at very low conf)
- empty_box: not present in dataset

### Pipeline / script changes
- Initial `scripts/extract_frames.py`, `check_labels.py`, `split.py`, `review_app.py`
- 2-class config locked in `configs/data.yaml`
- Naming convention `<video_id>_fNNNN.jpg` enforced

### Known issues
- Model is not usable — too few training labels
- Single-video dataset means split is leak-prone
- Triage CSV initially saved to wrong path (`dataset/_raw/train.txt`); fixed by archiving

---

## [pipeline] 2026-05-01 — video traceability + QC gate

Not a model release; pipeline upgrades while staging more data.

### Pipeline / script changes
- `extract_frames.py`: `--video-id` flag, auto-derive from filename, validation
- `split.py`: removed per-image fallback; now requires ≥2 video_ids; group strictly by `video_id`
- `review_app.py`: header shows `Video: <id> — Frame: <fNNNN>`, sidebar filter by `video_id`, group navigation buttons, CSV adds `video_id` column
- New: `scripts/stage_to_label.py` (focused labeling subset, ≤10 per video, evenly spaced)
- New: `scripts/merge_labels.py` (push `_to_label/` back into `_raw/`)
- New: `scripts/qc_video.py` (rule-based video gate; critical: too_short/no_motion/no_signal/blurry)
- Streamlit image render capped at 60vh + 720px max

### Dataset state at end of day
- 5 video_ids extracted: `TTKMFB581` (14), `ORD854132753835` (63), `ORD854132168226` (53), `TTKMFB583013521` (48), `TTKMFB583030068` (67) = 245 frames
- Triage complete: 81 yes / 147 no across all 5 videos
- 43 frames staged in `dataset/_to_label/` for labeling

### Next milestone (target v2)
- Finish labeling 43 staged frames
- Retrain at 30 epochs
- Target mAP@50 > 0.4 on val (now properly stratified across 5 videos)

---

## [pipeline] 2026-05-01 (later) — Streamlit dashboard, multi-user, versioned datasets

Major UI/architecture pass. No model training yet — tooling refresh.

### New & updated scripts
- `scripts/pipeline_runner.py` (NEW): non-blocking QC + shipping_label/OCR + damage orchestrator. `run_pipeline(video)` returns a structured result dict with per-module signals + `final_status` (AUTO_PASS / REVIEW_REQUIRED / NO_DAMAGE / INVALID_INPUT). Module failures become signals, never halt
- `scripts/pipeline.py`: superseded by pipeline_runner; kept for reference
- `scripts/review_app.py`: rewritten as a 4-tab dashboard
  - **Review labels**: existing yes/no triage, now per-task + per-version
  - **Import frames**: 3-step Extract → multi-select → Send (creates new version)
  - **Inspect & label**: read-only inspector + LabelImg launch button + per-frame queue + per-user assignments + carry-forward
  - **Run pipeline**: video uploader → pipeline_runner → structured display + AI-hint overlay
- `scripts/extract_frames.py`: added `output_dir` parameter (CLI behavior unchanged)
- `dataset/<task>/<base>/v<date>_<note>/` versioned subfolder convention. `_meta.json` per version records fps, frames_by_video, created_at. `v_legacy` = pre-versioning data at the base folder (no file moves)
- `<version_dir>/_assignments.json`: per-version filename → user mapping for multi-user labeling
- `outputs/pipeline/<video_id>.json`: pipeline_runner output
- `outputs/uploads/`: Streamlit upload landing zone

### New dependencies
- `easyocr==1.7.2` (English OCR for tracking codes; --no-deps to avoid opencv-python-headless conflict)
- `streamlit-autorefresh==1.0.1` (5-second filesystem-poll opt-in)
- `streamlit-drawable-canvas==0.9.3` (installed but unused after C3 decision; kept in venv harmlessly)

### LabelImg integration
- PyQt5 float-type patches at `libs/canvas.py:526,530-531` and `libs/shape.py:131` (int()-wrap on QPainter coords) — required for newer PyQt5 versions
- Streamlit "🚀 Open in LabelImg" button → detached subprocess at the active image; auto-refresh detects saved `.txt` files and flips badges 🟡 → 🟢
- Inspector is read-only by design; the in-browser drawing experiment was abandoned (streamlit-drawable-canvas can't unify draw + select/edit modes; no maintained alternative on PyPI)

### Multi-user / safety guardrails
- Sidebar `Current user` selector (USERS = ["user_a", "user_b", "user_c", "user_d"]) — replace with real auth in production
- Atomic JSON writes for assignments (file-locking deferred until needed)
- Double-Send guard (10s window): warning + Reuse latest version (only if labels-free) / Create-anyway buttons
- Active-version switch warning when leaving in-flight queue/active state
- Carry-forward FPS-mismatch detection + Safe-mode (cached MD5 verification) for label transfer between versions

### Dataset state at end of session
- Damage: 5 weak labels on 1 video (`TTKMFB581`) — no change since v1
- Shipping Label: 4 video_ids (`TTKMFB-583030068944012504`, `TTVN1060812452`, `TTVN1069599748`, `TTVN1066684755`), 77 frames staged in `_to_label/`, 2 labels (TTKMFB-...f0001/f0002)

### Known gaps before next training run
- Need **~28 more shipping_label labels** before any training. Path: open Streamlit Inspect tab → 🚀 Open in LabelImg → label 25–30 frames across the 4 videos
- `dataset/_import/` may have orphan frames from earlier flow iterations — manually clean if it matters
- Filename contract still `<video_id>_f<NNNN>.jpg` (timestamp-filename change deferred — would require updating `parse_video_id` in qc_video.py + check_tracking.py + split.py)

---

## [v2] 2026-05-02 — first shipping_label model + production-ready dashboard

First end-to-end shipping_label training run + a dense layer of dashboard guardrails. Damage stays at v1 (still 5 labels, no work done there this session).

### Model
- Run name: `rl_shipping_label_v2026-05-01_40659` (auto-suffix: empty version-note → `v<date>_<5digit_hash>`)
- Base weights: `yolov8n.pt`
- Single class: `shipping_label`
- Trained on the v_legacy slice of `dataset/shipping_label/_to_label/` after `split.py` (4 train videos / 1 val video / 11 labels). Negatives: 112 unlabeled frames included
- Weights at `models/rl_shipping_label_v2026-05-01_40659/weights/best.pt`
- Status: smoke-test only — 11 labels is well below useful coverage; this is a "the pipeline trains end-to-end" milestone, not a deployable model

### Pipeline / script changes (Streamlit dashboard)
- **Train tab** (NEW, 5th tab): non-blocking subprocess training via `scripts/_train_runner.py` (NEW); `logs/train_<task>_<version>.log` + `.state.json` per task+version. Live source counts (root .jpg/.txt) vs last-split snapshot, with drift warning + 🔁 Re-split now button. Training guard: warns if <10 labels or >50% excluded
- **Versioned datasets** (settled): `dataset/<task>/<base>/v<date>_<note>/`. Per-version `_data.yaml` auto-generated by Train tab — `path:` is absolute, classes from `load_class_names`. Per-version `_assignments.json` and `_meta.json`
- **Versioning guardrails**: named versions (`v<YYYY-MM-DD>_<note>`), Vietnamese-aware note sanitization (NFKD + accent strip + `[a-z0-9]` slug), preview before commit, 10s double-Send guard, active-version switch warning, sidebar `Show archived versions` toggle
- **Append vs Create new** (Send mode): default Append preserves session state (queue/active/filters); Create new switches version + resets context. Two helper functions: `_append_send_block` / `_create_new_version_block`
- **Carry-forward labels**: 📦 Copy labels from a previous version. FPS-mismatch detection (red error + required confirmation). Safe-mode (cached MD5 byte-equality check) auto-on when fps is missing/mismatched. Reports `{copied, skipped_existing, no_match_in_source, skipped_hash_mismatch}`
- **Multi-user**: per-(task, version, user) session_state isolation. Sidebar `Current user` selector. `_excluded.json` per-version with per-frame `Use for training` toggle + bulk `🚫 Exclude` from queue
- **Auto-OCR pipeline** (NEW): `<frame>.ocr.json` sidecar with auto + corrected layers. Triggered each Inspect-tab render for any labeled frame whose `.txt` is newer than its sidecar. Multi-rotation (0/90/180/270) — picks highest-confidence rotation. User corrections editable per box, never overwritten by re-OCR. Per-task validation rule (shipping_label: `^[A-Z]{2,8}[0-9]{6,18}$`)
- **Strict pipeline contract** (per spec): `run_shipping_label_detection` runs explicitly BEFORE OCR. Weights auto-resolved to latest `models/rl_shipping_label_*/weights/best.pt` (overridable via dropdown). conf_threshold = 0.1 (debug). NO fallback OCR — `no_label_detected` short-circuits. UI shows red bbox on best-confidence frame + detection log; falls back to red 🚫 NO LABEL DETECTED banner
- **Pipeline tab cache invalidation**: `(absolute_path | mtime)` signature key. Same-filename re-uploads with different bytes auto-invalidate
- **`split.py`**: `--data` flag for task switching, auto-detects source (`_raw` / `_to_label` / version root). Honors `<source>/_excluded.json`
- **`train.py`**: `--data` flag, `_print_dataset_debug()` shows abs paths + counts before training
- **`extract_frames.py` `derive_video_id`**: NFKD-aware, replaces spaces+special chars with `-` (preserves case). `"VD KH TTKMFB-583030068944012504.mp4"` → `"VD-KH-TTKMFB-583030068944012504"` instead of just `"VD"`
- **Square thumbnails**: Inspect/Pool grids use center-cropped 240×240 / 200×200 squares. Equal row heights, no layout shift
- **Pool video filter**: search-by-text input above the dropdown for substring match across video_ids

### New scripts
- `scripts/_train_runner.py`: detached wrapper running `split.py` → `train.py`, writes JSON state file
- `scripts/check_dataset.py`: pre-train dataset sanity check; flags orphan labels and empty `.txt`. "Image without label" is INFO (YOLO negatives), not an error

### LabelImg gotchas resolved
- 4 PyQt5 float-type patches (canvas.py:526, 530-531; shape.py:131)
- Detached `subprocess.Popen` for cross-tab launch
- `🔄 Auto-refresh (5s)` checkbox in Inspect tab uses `streamlit-autorefresh`
- Streamlit checkbox bulk-action footgun documented in memory (`reference_streamlit_checkbox_bulkops.md`) — bulk operations must write through to widget keys, not just the shadow set

### Dataset state at end of session
- Shipping Label: **5 video_ids** (added `TTVN1064995559`), **123 frames** in `_to_label/`, **11 labels** (10 train + 1 val after split). val held-out video: `TTVN1066684755`
- Damage: still 5 weak labels on `TTKMFB581` only — no work this session
- Trained model: `rl_shipping_label_v2026-05-01_40659` (smoke-test only)

### Known gaps / next session
- Val set has 1 label — mAP@50 will be statistical noise. Label ~5 more frames in `TTVN1066684755` before treating numbers as meaningful
- f0001's existing label is suspect (auto-OCR returned conf 0.222 garbage `'eene Eeese 1...'`) — likely drawn around the barcode instead of the tracking text. Re-label
- Need ~20 more shipping_label labels for a model that generalizes
- `streamlit-drawable-canvas` still installed but no longer used (in-browser labeling abandoned). Keep for now; small disk footprint
- AI-hint center overlay in `_render_result` is dormant under strict pipeline mode — consider removing in next cleanup pass
