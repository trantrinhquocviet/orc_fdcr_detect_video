---
title: ORC FDCR Detect Review
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# Reverse-Logistics Local AI (YOLOv8 + Python)

Local desktop pipeline that analyzes return-package videos/images for two
business signals: **damaged_item** and **empty_box**, plus a separate
**shipping_label** detector for tracking-code OCR. No cloud calls. Built for
fast iteration on small datasets.

The primary UI is the **Streamlit dashboard** (`scripts/review_app.py`) —
4 tabs covering import, labeling, inspection, and full pipeline runs.
Individual CLI scripts still exist as fallbacks.

> Status: MVP. Pipeline runs end-to-end. Model accuracy depends entirely on
> how many frames you've labeled — see *Iteration loop* below.

---

## 1. Project structure

```
ORC detect video/
├── configs/
│   ├── data.yaml                 # YOLO dataset config — Damage task (2 classes)
│   └── predefined_classes.txt    # Class list for LabelImg — Damage task
├── dataset/
│   ├── _import/                  # neutral pool from Import tab Stage 1
│   ├── _raw/                     # Damage task base. Has v_legacy data + v* subfolders
│   │   ├── *.jpg                 #   v_legacy data lives here (pre-versioning)
│   │   ├── _assignments.json     #   per-version sidecar (filename → user)
│   │   └── v<YYYY-MM-DD>_<note>/ #   versioned subfolders; each has _meta.json
│   ├── shipping_label/           # Shipping label task (separate model)
│   │   ├── _to_label/            #   base; same versioning convention as _raw/
│   │   └── data.yaml             #   per-task YOLO config
│   ├── images/{train,val}/       # populated by split.py
│   └── labels/{train,val}/       # YOLO .txt files, populated by split.py
├── models/                       # training runs land here; weights/best.pt is what to ship
├── outputs/
│   ├── qc/<video_id>.json        # per-video QC verdicts
│   ├── pipeline/<video_id>.json  # pipeline_runner non-blocking results
│   ├── uploads/                  # videos uploaded via Streamlit
│   ├── frames/<video_stem>/      # per-video frame extraction (from infer.py)
│   ├── json/<file_stem>.json     # detection results from infer.py
│   ├── annotated/                # debug images with boxes drawn
│   ├── tracking/                 # check_tracking.py outputs
│   └── review/                   # Streamlit review CSVs
├── scripts/
│   ├── review_app.py             # Streamlit dashboard — 5 tabs (PRIMARY UI)
│   ├── pipeline_runner.py        # non-blocking QC + tracking + damage orchestrator
│   ├── pipeline.py               # superseded by pipeline_runner.py
│   ├── _train_runner.py          # detached wrapper: split.py → train.py with state JSON
│   ├── qc_video.py               # rule-based video gate
│   ├── check_tracking.py         # shipping_label detection + EasyOCR (strict mode)
│   ├── extract_frames.py         # video → frames at fixed FPS
│   ├── stage_to_label.py         # build _to_label/ from triage CSV
│   ├── check_labels.py           # pre-train QC on labels
│   ├── check_dataset.py          # pre-train dataset sanity (orphan labels, empty .txt)
│   ├── merge_labels.py           # push _to_label/ labels back to _raw/
│   ├── split.py                  # video-stratified train/val split (versioned)
│   ├── train.py                  # YOLOv8 training (versioned)
│   ├── infer.py                  # one-shot CLI (image / video / folder)
│   ├── serve.py                  # long-running stdin/stdout JSON-RPC for Electron
│   └── rules.py                  # business rule engine on top of detections
├── logs/                         # train_<task>_<version>.log + .state.json
├── requirements.txt
└── README.md
```

---

## 2. Setup (Windows)

```bash
# Python 3.10–3.12 recommended
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install labelImg streamlit
```

Optional but recommended: install **FFmpeg** for system-wide video work.
`extract_frames.py` and `qc_video.py` use OpenCV directly so they don't
require FFmpeg.

> All commands below assume the venv is the active interpreter. From bash on
> Windows you can also run them via the explicit path:
> `.venv/Scripts/python.exe scripts/<name>.py`

---

## 3. Classes (locked at 2 for MVP)

| ID | Class | Definition |
|---|---|---|
| 0 | `damaged_item` | Visible tear, hole, dent, leak, exposed contents — box the **damage region only**, not the whole package and not hands/tools |
| 1 | `empty_box` | Open container clearly empty (or only filler, no product) |

If the frame is borderline, **skip it**. Do not pollute the dataset with
hedged labels.

---

## 4. Streamlit dashboard (primary UI)

```bash
.venv/Scripts/streamlit.exe run scripts/review_app.py
```

Sidebar selectors (all four tabs read these): `Training task` (Damage /
Shipping Label) · `Dataset version` (newest first; `v_legacy` = pre-versioning
data) · `Current user` (simulated team — replace with auth in production).

| Tab | Purpose |
|---|---|
| **Review labels** | Yes/no triage on existing labels. Outputs CSV → `outputs/review/review_<task>_<version>.csv` |
| **Import frames** | 3-step flow. (1) Extract video → `dataset/_import/`. (2) Per-frame multi-select with video_id stamps + search-by-text. (3) Send modes: **Append** (default — preserves session state, adds to current version) or **Create new version** (`v<YYYY-MM-DD>_<note>`, switches sidebar). Hard-link or copy. Writes `_meta.json` with fps + provenance |
| **Inspect & label** | Active-version banner + 4-stat summary · filters (Status / Video / Assignment) · 4-col grid with status (🟢/🟡), assignment (👤/🆓/👥), and exclude (🚫) badges · 240×240 square thumbnails (no row jitter) · `🚀 Open in LabelImg` button (launches detached subprocess at the active image) · `🔄 Refresh` + `Auto-refresh (5s)` · multi-image queue with `📋 Queue` checkboxes · per-frame `✅ Use` (inverse of exclude) toggle · `👤 Claim N unassigned` · `📦 Copy labels from previous version` (with FPS-mismatch + Safe-mode MD5 verification) · **auto-OCR** on labeled boxes (multi-rotation, sidecar `<frame>.ocr.json`) with editable corrections preserved across re-runs |
| **Train** | Non-blocking subprocess training. Per-task per-version: `🚀 Retrain <task> · <version>`, `🔁 Re-split now (no train)`, `🔄 Refresh`. Live source counts (root .txt) vs last-split snapshot with drift indicator. Training guard warns if `<10` labels or `>50%` excluded. Logs: `logs/train_<task>_<version>.log`. State: `.state.json` with status / step / exit_code / pid (stale after 5 min). Dataset management expander: 🧹 Clean dataset (creates v<date>_clean), 📦 Archive version |
| **Run pipeline** | Upload/path → runs `pipeline_runner.run_pipeline()` with auto-resolved `models/rl_shipping_label_*/weights/best.pt` (latest by mtime) → explicit `run_shipping_label_detection` BEFORE OCR · OCR runs ONLY on detected box crops (no fallback) · pattern validation `^[A-Z]{2,8}[0-9]{6,18}$` · UI shows red bbox on best-confidence frame OR red 🚫 NO LABEL DETECTED banner. Cache invalidated by `(path, mtime)` signature so re-uploads never show stale results |

### Key invariants

- **Versioned datasets**: every Send creates a new `v<date>_<note>/`. Old labels stay in their original version. `_meta.json` records `fps`, `frames_by_video`, `created_at`. Re-extracting at a different fps is safe — goes to a separate version.
- **Per-(task, version, user) state isolation**: every session_state key is scoped, so switching task/version/user preserves independent queues, active selections, and filter prefs.
- **File-based assignments**: `<version_dir>/_assignments.json` (atomic writes). Anyone can claim from the unassigned pool. No two users see the same `🆓` frame as available — once claimed, it shows `👥` to others.
- **Carry-forward safety**: FPS mismatch shows a red error + required confirmation. Safe-mode (default-on when FPS missing) hashes both source and destination images, copies `.txt` only when bytes are byte-identical.

The CLI scripts in §5 below are still functional and useful as a fallback or for batch jobs. The dashboard wraps the labeling-loop subset; CLI handles training and ad-hoc runs.

---

## 4b. End-to-end pipeline (CLI loop, fallback)

```
   videos
     │
     ▼
[1] qc_video.py      ──────────►  reject low-quality videos before doing real work
     │ (PASS only)
     ▼
[2] extract_frames.py            video → dataset/<task>/<base>/v<...>/<video_id>_fNNNN.jpg
     │
     ▼
[3] review_app.py     ──────────► Inspect & label tab → 🚀 Open in LabelImg per frame
     │                            (or LabelImg directly via CLI, see §5.5)
     ▼
[4] check_labels.py              QC: bad class IDs, tiny/huge boxes, etc.
     │
     ▼
[5] split.py                     video-stratified split (≥2 video_ids required)
     │
     ▼
[6] train.py                     train YOLOv8 → models/<name>/weights/best.pt
     │
     ▼
[7] infer.py                     run trained model on new videos/images
     │ (or pipeline_runner.py for QC + tracking + damage all-in-one)
     ▼
[8] review_app.py     ──────────► Review labels tab — mark ✅/❌ → CSV
     │
     ▼
   improve loop: add more frames → relabel weak cases → retrain (new version each time)
```

---

## 5. Quick reference — every command

All commands assume `cd "C:/Users/OP-LT-0496/Downloads/ORC detect video"`.

### 5.1 Gate a new video (QC)

```bash
.venv/Scripts/python.exe scripts/qc_video.py --input "path/to/video.mp4"
```

Exit code `0` = PASS, `1` = FAIL. Reasons saved to `outputs/qc/<video_id>.json`.

Critical rules: duration < 8s, no motion, no detections, mostly blurry. Warning
rules: too dark, occluded, bad framing.

### 5.2 Extract frames

```bash
.venv/Scripts/python.exe scripts/extract_frames.py \
  --video "path/to/video.mp4" \
  --video-id ORD854132753835 \
  --fps 1
```

`--video-id` auto-derives from filename if omitted. Filename rule (enforced):
`<video_id>_fNNNN.jpg`. video_id must be alphanumeric + dashes, no underscores.

### 5.3 Triage in Streamlit

```bash
.venv/Scripts/streamlit.exe run scripts/review_app.py --server.headless true
```

Browser → `http://localhost:8501`. Uncheck "Show only images with labels"
(sidebar) when triaging unlabeled frames. CSV → `outputs/review/review_<folder>.csv`
with `timestamp, video_id, image_name, predicted_label, is_correct`.

### 5.4 Stage focused labeling folder

```bash
.venv/Scripts/python.exe scripts/stage_to_label.py
```

Reads `outputs/review/review__raw.csv`, picks up to 10 yes-frames per video
(evenly spaced), copies to `dataset/_to_label/`. Existing `_raw/*.txt` labels
are copied too so LabelImg shows current boxes.

### 5.5 Label in LabelImg

```bash
.venv/Scripts/python.exe .venv/Lib/site-packages/labelImg/labelImg.py \
  dataset/_to_label configs/predefined_classes.txt dataset/_to_label
```

In the GUI:
- Toggle format → **YOLO** (left toolbar)
- `W` = draw box, `Ctrl+S` = save, `D` = next, `A` = previous
- `Ctrl + scroll` = zoom; `Ctrl+F` = fit window
- View → enable **Auto Save Mode** if available

### 5.6 Merge labels back

```bash
.venv/Scripts/python.exe scripts/merge_labels.py
```

Copies every `*.txt` from `_to_label/` into `_raw/`, overwriting older labels.

### 5.7 Pre-train QC

```bash
.venv/Scripts/python.exe scripts/check_labels.py
```

Flags missing/empty labels, bad class IDs (must be 0 or 1), out-of-range
coords, suspiciously tiny or huge boxes. Prints per-class instance counts.

### 5.8 Train/val split

```bash
.venv/Scripts/python.exe scripts/split.py
```

Strict video-stratified split (no leakage). Requires **≥ 2 video_ids**;
otherwise fails with exit code 2.

### 5.9 Train

```bash
.venv/Scripts/python.exe scripts/train.py \
  --model yolov8n.pt --epochs 30 --imgsz 640 --batch 4 --device cpu --name rl_yolov8n_v2
```

Run output lands in `models/<name>/`. Best weights at `weights/best.pt`,
ONNX export at `weights/best.onnx`.

### 5.10 Inference

```bash
# Image
.venv/Scripts/python.exe scripts/infer.py \
  --input "test.jpg" --weights "models/rl_yolov8n_v2/weights/best.pt"

# Video (sample 1 frame/sec)
.venv/Scripts/python.exe scripts/infer.py \
  --input "test.mp4" --weights "models/rl_yolov8n_v2/weights/best.pt" --fps 1 --device cpu

# Folder batch
.venv/Scripts/python.exe scripts/infer.py \
  --input "path/to/folder/" --weights "models/rl_yolov8n_v2/weights/best.pt"
```

JSON to `outputs/json/<stem>.json`, annotated images to `outputs/annotated/`.

### 5.11 Review predictions

Same Streamlit app. Point the sidebar "Image folder" at any folder with
matching image + `.txt` pairs (e.g. `outputs/annotated/<video_stem>/` after
`yolo predict --save-txt`).

---

## 6. File-naming convention (enforced)

```
<video_id>_fNNNN.jpg
<video_id>_fNNNN.txt   (matching label, same stem)
```

- `video_id`: alphanumeric + dashes only. No underscores (collides with `_f`).
- `NNNN`: 4-digit zero-padded frame index (1-based).

Why this matters:
- Lets `split.py` group by video without leakage
- Lets `review_app.py` filter by `video_id` and group-navigate
- Traces every frame back to the source case for ops review

---

## 7. Output formats

### 7.1 QC verdict (`outputs/qc/<video_id>.json`)

```json
{
  "video": "path/to/video.mp4",
  "video_id": "ORD854132753835",
  "status": "PASS",
  "reasons": [],
  "metrics": {
    "duration": 62.77,
    "fps": 29.97,
    "total_frames": 1881,
    "frames_sampled": 10,
    "brightness": 126.68,
    "motion_score": 57.334,
    "blur_score_mean": 421.2,
    "blurry_frame_ratio": 0.2,
    "detections": 4,
    "largest_box_area_rel": 0.18,
    "mean_box_area_rel": 0.07
  }
}
```

Critical reasons that flip status to FAIL: `too_short`, `no_motion`,
`no_signal`, `blurry`. Warning reasons (still listed but don't fail):
`too_dark`, `occluded`, `bad_framing`.

### 7.2 Inference output (`outputs/json/<stem>.json`)

```json
{
  "file": "video.mp4",
  "type": "video",
  "verdict": "high_risk",
  "issues_detected": ["high_risk", "damage"],
  "issue_counts": {"high_risk": 3, "damage": 1},
  "frames_flagged": ["frame_00003.jpg"],
  "confidence": 0.87,
  "frame_count": 24,
  "frames_sampled_fps": 1.0,
  "per_frame": [
    {
      "frame": "frame_00001.jpg",
      "detections": [
        {"class": "damaged_item", "confidence": 0.91, "bbox": [12.0, 33.5, 540.1, 410.2]}
      ],
      "max_conf": 0.91,
      "issues": ["damage"]
    }
  ],
  "elapsed_sec": 3.214
}
```

### 7.3 Review CSV (`outputs/review/review_<folder>.csv`)

```
timestamp,video_id,image_name,predicted_label,is_correct
2026-05-01T00:06:42,TTKMFB581,TTKMFB581_f0004.jpg,damaged_item,yes
```

---

## 8. Rule engine (`scripts/rules.py`)

Maps raw detections → business issues:

- `empty_box` → **missing_item**
- `damaged_item` → **damage**

(The original 5-class scheme also defined `high_risk` from `open_box + no_seal`.
Those classes are out of scope for the 2-class MVP; the unused branches in
`rules.py` go silent. Re-enable them when adding more classes.)

The file-level `verdict` picks the most severe issue across flagged frames.
Edit `SEVERITY` in `rules.py` to retune priority for your ops team without
touching the detector.

---

## 9. Iteration loop — how to improve the model

After each training run, look at exactly two things:

1. `models/<name>/results.png` — loss curves should trend down, mAP should trend up
2. `models/<name>/val_batch0_labels.jpg` vs `val_batch0_pred.jpg` — visual sanity check

Then pick the **single worst symptom** and fix only that:

| Symptom | Fix |
|---|---|
| One class never detected | Add 15+ frames of that class, varied conditions |
| Many false positives on intact items | Add 10 negative frames (no labels, just images) |
| Boxes too loose / too tight | Re-label your 10 worst training images |
| Confuses classes | Re-read class definitions; re-label edge cases |
| mAP@50 < 0.3 after 2 retrains | Stop tweaking — labels are the problem, audit them |

Stop when **mAP@50 > 0.6 on a properly stratified val set with ≥ 4 videos**
and ops reviewers say the predictions look right. Ship that as MVP v1.

---

## 10. Performance levers (when you need them)

Already wired in:
- Batch inference (`--batch 16` is one forward pass for 16 frames)
- Frame skipping (`--fps 1` samples 1 frame/sec instead of 30)
- Lightweight model (yolov8n, ~6 MB, ~10–15 ms/frame on CPU)

Further wins:
- Export to ONNX/TensorRT (`train.py` already exports ONNX)
- Half precision on supported GPUs: `model.predict(..., half=True)`
- Resolution downshift: `--imgsz 480` halves compute
- Two-stage filter for high-volume: cheap "any issue?" classifier first,
  then full detector only on flagged frames

---

## 11. Electron integration

Two patterns:

### Recommended: Python sidecar (low latency)

`serve.py` runs as a long-lived subprocess and exchanges newline-delimited
JSON with the Electron main process. Bundle Python + serve.py via PyInstaller
(`pyinstaller --onefile scripts/serve.py`) and ship in `extraResources`.

### Simpler: per-call CLI (higher latency)

Spawn `python scripts/infer.py --input ... --quiet` per analysis. Cold-start
~3–5s per call — fine for occasional manual review, painful for batch QC.

---

## 12. Scaling to 50k+ videos / month

50,000 × 30s × 1 fps = 1.5M frames/month. Local desktops can't sustain this:

- **Worker pool**: containerize `serve.py` (one process per GPU), front it
  with a queue (Redis Streams, RabbitMQ). Workers pull videos, write JSON to
  object storage.
- **GPU sizing**: yolov8n at 640px on a single T4 ≈ 120 fps batched; one T4
  comfortably handles 1.5M frames/month.
- **Two-stage filter**: 5-class binary "any-issue?" classifier at 320px
  drops ~70% of frames cheaply; full detector on the rest.
- **Storage**: keep raw frames only for flagged videos. Plain `verdict=ok`
  → JSON + thumbnail; discard frames after 7 days.
- **Active learning**: ops reviewers correct the 1–2% of borderline cases
  via the Streamlit app; corrections feed straight back into `dataset/_raw/`
  for the next training run.

---

## 13. Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install -r requirements.txt` |
| LabelImg won't launch from `.exe` | Use `python .venv/Lib/site-packages/labelImg/labelImg.py ...` |
| LabelImg saves `.xml` not `.txt` | Toggle format button (left toolbar) until it shows YOLO |
| `split.py` fails with "only 1 video" | Add a 2nd video via `extract_frames.py` |
| `check_labels.py` flags out-of-range coords | Open file in a text editor; values must be 0–1 |
| `qc_video.py` always says `no_signal` | Weights are weak (small dataset) — retrain with more labels |
| Streamlit image too tall on screen | Already capped at 60vh + 720px max; refresh the page |

---

## 14. Quick smoke test (no trained weights)

```bash
.venv/Scripts/python.exe scripts/infer.py --input any.jpg --weights yolov8n.pt --conf 0.25
```

Uses stock COCO weights — won't detect `damaged_item` etc., but proves the
pipeline runs end-to-end and produces valid JSON.
