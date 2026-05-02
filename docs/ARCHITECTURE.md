# Architecture — return-detection MVP

For engineers picking this up. Covers data flow, module boundaries,
integration surfaces, and the deliberate trade-offs.

---

## 1. System context

Single-machine Python pipeline. No services, no DB, no cloud calls. Designed
to be embedded in a future Electron desktop app via a Python sidecar.

```
   warehouse                ┌─────────────────────────┐
   videos (mp4) ─┐          │  Streamlit dashboard    │
                 │          │  (scripts/review_app.py)│
   warehouse     │  files   │  5 tabs · primary UI    │
   photos (jpg) ─┤◄────────►└────────────┬────────────┘
                 │                       │
                 │                       │ subprocess (Train tab)
                 │                       │ ─► _train_runner.py ─► split.py + train.py
                 │                       │
                 │                       │ in-process (Run pipeline tab)
                 │                       │ ─► pipeline_runner.run_pipeline()
                 │                       │
                 │                       ▼
                 │          ┌─────────────────────────┐
                 │          │  Ultralytics YOLOv8     │
                 │          │  (CPU, per-task model)  │
                 │          └────────────┬────────────┘
                 │                       │
                 │                       ▼
                 │          ┌─────────────────────────┐
                 │          │  rules.py + easyocr     │
                 │          │  (issue logic + OCR)    │
                 │          └────────────┬────────────┘
                 │                       │
                 │                       ▼
                 │          ┌─────────────────────────┐
                 └─────────►│  outputs/{pipeline,json,│
                            │  qc,review}/...         │
                            └─────────────────────────┘

           future:  Electron ──► scripts/serve.py (long-lived JSON-RPC sidecar)
```

Two pieces of cv weight are loaded by the runtime:
- A **damage** model — 2 classes (`damaged_item`, `empty_box`)
- A **shipping_label** model — 1 class (`shipping_label`), feeds OCR for tracking-code text

---

## 2. Two distinct loops

### 2.1 Labeling + training loop (dashboard-driven)

```
videos
  │
  ▼
qc_video.py ──► PASS? ──no──► reject + log              [Pipeline tab gates fresh inputs]
  │ yes
  ▼
extract_frames.py ──► dataset/_import/<vid>_fNNNN.jpg   [Import tab Step 1]
  │
  ▼
review_app.py: pick frames + Send mode ──┐              [Import tab Step 2-3]
                                         │
                  ┌──────────────────────┴──────────────────┐
                  ▼                                          ▼
       Append to current version            Create new version
       <task_dir>/<version>/<vid>_fNNNN.jpg  <task_dir>/v<date>_<note>/...
                  │                                          │
                  └──────────────┬───────────────────────────┘
                                 ▼
       LabelImg (subprocess from Inspect tab)
       ──► <version_dir>/<vid>_fNNNN.txt
                                 │
                                 ▼
       Auto-OCR pass on labeled boxes (multi-rotation)
       ──► <version_dir>/<vid>_fNNNN.ocr.json   {auto + corrected layers}
                                 │
                                 ▼
       Train tab: 🚀 Retrain <task> · <version>
       ──► _train_runner.py: split.py → train.py
       ──► models/rl_<task_slug>_<version>/weights/best.pt
```

### 2.2 Inference loop (pipeline_runner — non-blocking, signal-aggregating)

```
new video
  │
  ▼
pipeline_runner.analyze(video, qc_w, tracking_w, damage_w):
  │
  ├─► run_qc(video, qc_w)             ──► {qc_status, qc_reasons}
  │   (qc_video.qc_video — never gates next steps)
  │
  ├─► run_shipping_label_detection(video, tracking_w, conf=0.1)  [NEW: explicit pre-OCR step]
  │   ├─► auto-resolves models/rl_shipping_label_*/weights/best.pt
  │   ├─► samples 12 frames evenly + YOLO at conf 0.1
  │   ├─► logs: model path · classes · per-frame confidences
  │   └─► returns: {status, frames_sampled, detections_count, confidences, best_idx, ...}
  │
  ├─► run_tracking(video, tracking_w):
  │   ├─► if detection.status != "detected" → no_label_detected, NO OCR
  │   └─► else check_tracking.check_tracking(strict_no_box_fallback=True):
  │       ├─► OCR on box CROPS only (no full-frame fallback)
  │       └─► validate text vs ^[A-Z]{2,8}[0-9]{6,18}$ pattern
  │   ──► {visible, valid, text, status, detection: {...}}
  │
  └─► run_damage(video, damage_w)     ──► {damage_detected, count, confidence}
      (infer.analyze_file + rules.py)
                                          │
                                          ▼
                          decide(tracking_valid, damage_detected):
                          AUTO_PASS / REVIEW_REQUIRED / NO_DAMAGE / INVALID_INPUT
                                          │
                                          ▼
                          outputs/pipeline/<video_id>.json + UI display with red bbox overlay
```

**Non-blocking contract**: any module failure becomes a signal in the result. The pipeline never halts. Each module returns its own structured signal block.

---

## 3. Module responsibilities

| Module | Single responsibility | Reads | Writes |
|---|---|---|---|
| `qc_video.py` | Reject low-quality videos before any heavy work | video file | `outputs/qc/<vid>.json` |
| `extract_frames.py` | Sample frames at fixed FPS, enforce naming, NFKD-aware video_id derivation | video file | configurable `<output_dir>/<vid>_fNNNN.jpg` |
| `check_tracking.py` | YOLO `shipping_label` detection + EasyOCR. **Strict mode** (default in pipeline_runner) skips OCR when no box; tracking_pattern validates text | video + weights | `outputs/tracking/debug/<vid>.json` (when `save_debug=True`) |
| `check_labels.py` | Validate label sanity before training | dataset folder | stdout report |
| `check_dataset.py` (NEW) | Pre-train pair-wise sanity. "Image without label" treated as info (YOLO negative), not an error | data.yaml or dataset folder | stdout |
| `merge_labels.py` | Push staged `.txt` back into canonical store (legacy flat layout) | `_to_label/*.txt` | `_raw/*.txt` |
| `split.py` | Video-stratified train/val sets. Honors `<source>/_excluded.json`. `--data` arg takes any data.yaml; auto-detects source folder | data.yaml + source dir | `<data_root>/{images,labels}/{train,val}/` |
| `train.py` | Fine-tune YOLOv8. `--data` arg + `_print_dataset_debug` shows abs paths before YOLO starts | data.yaml + dataset | `models/<run>/...` |
| `infer.py` | Run a single trained model + rule engine on input | weights + image/video/folder | `outputs/{json,annotated}/...` |
| `pipeline_runner.py` (NEW) | Non-blocking orchestrator: QC + tracking + damage. `run_shipping_label_detection` runs explicitly BEFORE OCR. `resolve_shipping_label_weights` auto-picks the latest model | video + weights | `outputs/pipeline/<vid>.json` |
| `serve.py` | Long-lived sidecar for Electron (single-model JSON-RPC) | stdin JSON | stdout JSON |
| `rules.py` | Map raw classes → business issues | detection list | issue list |
| `review_app.py` | Streamlit dashboard — 5 tabs (Review labels / Import frames / Inspect & label / Train / Run pipeline) | dataset folders + sidecars + weights | sidecars + CSVs + spawn LabelImg / training subprocess |
| `_train_runner.py` (NEW) | Detached subprocess wrapper for `split.py` → `train.py`. Writes `logs/train_<task>_<version>.{log,state.json}` | data.yaml | logs + state file |

**Boundary rule**: every script either reads stdin/files OR writes stdout/files, never both arbitrarily. Lets us swap any step without breaking the others.

---

## 4. Filename contract (the load-bearing convention)

All frames must be named `<video_id>_fNNNN.jpg`.

- `video_id`: alphanumeric + dashes only, no underscores
- `_f` is the parser delimiter
- `NNNN`: 4-digit zero-padded 1-based index

This single convention powers:

- **`split.py`** groups by `video_id` to prevent train/val leakage
- **`review_app.py`** filters by `video_id` and supports group navigation
- **`outputs/qc/<video_id>.json`** keys verdicts to source video
- **traceability**: every result row in any CSV/JSON can be traced to one source video

`extract_frames.py` enforces it (`derive_video_id` is now NFKD-aware: handles spaces and Vietnamese accents by replacing with `-`); `split.py` skips and logs malformed names. The `<video_id>` field is NEVER allowed to contain `_` because that collides with `_f`.

---

## 5. Data layout (versioned)

Every task directory holds one or more **dataset versions**. A version is either the legacy base folder (`v_legacy`) or a real subfolder named `v<YYYY-MM-DD>_<note>` (e.g. `v2026-05-01_clean`). Each version is fully self-contained: separate split, separate YOLO config, separate weights output.

```
dataset/
├── _import/                                    # neutral pool (Stage 1 of Import tab)
│   └── <video_id>_fNNNN.jpg
│
├── _raw/                                       # Damage task base
│   ├── <video_id>_fNNNN.jpg                   # v_legacy data lives at root
│   ├── <video_id>_fNNNN.txt                   # YOLO label or absent (negative)
│   ├── predefined_classes.txt                 # LabelImg input
│   ├── _assignments.json                      # filename → user (per-version)
│   ├── _excluded.json                         # filenames skipped from training
│   ├── images/{train,val}/                    # populated by split.py
│   ├── labels/{train,val}/
│   └── v<YYYY-MM-DD>_<note>/                  # post-versioning subfolders
│       ├── <video_id>_fNNNN.jpg
│       ├── <video_id>_fNNNN.txt
│       ├── <video_id>_fNNNN.ocr.json          # auto-OCR sidecar (multi-layer)
│       ├── _meta.json                         # {task, created_at, fps, frames_by_video, ...}
│       ├── _data.yaml                         # auto-generated by Train tab (abs path)
│       ├── _assignments.json
│       ├── _excluded.json
│       └── images/{train,val}/ + labels/{train,val}/
│
└── shipping_label/_to_label/                   # Shipping Label task base
    └── (same structure as dataset/_raw/, plus data.yaml at the base)

outputs/
├── qc/<video_id>.json                          # one per QC'd video
├── pipeline/<video_id>.json                    # pipeline_runner result blocks
├── frames/<video_stem>/                        # extracted by infer.py (transient)
├── json/<file_stem>.json                       # detection results from infer.py
├── annotated/<file_stem>/                      # debug images with boxes drawn
├── tracking/debug/<video_id>.json              # check_tracking when save_debug=True
├── uploads/<video_name>                        # Streamlit-uploaded videos
└── review/                                     # Streamlit review CSVs (per task+version)

models/
└── rl_<task_slug>_<version>/                   # ultralytics layout, one dir per Train tab run
    └── weights/{best,last}.{pt,onnx}

logs/
└── train_<task_slug>_<version>.{log,state.json}  # Train tab subprocess output
```

**Source-of-truth rule**: the **version directory** is the truth. Everything in `images/`, `labels/`, `models/` is regenerable from it. The pool `_import/` is staging only.

**Per-version sidecars** (live next to the frames):

| File | Purpose |
|---|---|
| `_meta.json` | Provenance: created_at, fps, frames_by_video, source_pool, optional `archived: true` |
| `_data.yaml` | YOLO config, auto-written by Train tab. Absolute `path:` |
| `_assignments.json` | `{filename: user}` for multi-user labeling. Atomic JSON writes |
| `_excluded.json` | `[filename, ...]` — skipped by `split.py` and Clean dataset |
| `<frame>.ocr.json` | Per-image auto-OCR + corrections (preserved across re-OCR) |

---

## 6. Deliberate trade-offs

### Locked: 2-class damage + 1-class shipping_label (separate models)
- Shipping label was originally a 3rd class on the damage model; split into a separate single-class detector to avoid disturbing the locked damage model
- Pro: independent iteration, smaller class imbalance, OCR pipeline cleanly tied to one detector
- Con: two models to load at inference (~12 MB on CPU)
- Choice rationale: documented in memory under `project_decisions.md`

### Locked: video-stratified split, no per-image fallback
- Pro: val metrics are honest
- Con: fails when only 1 video uploaded
- Mitigation: `qc_video.py` and `extract_frames.py` make adding videos cheap

### Locked: versioned datasets (v_legacy + v<date>_<note>/)
- Pro: re-extracting at a different fps never corrupts old labels — they live in their original version
- Con: more disk; harder to "merge" labels across versions (mitigated by Carry-forward tool with FPS-mismatch + Safe-mode hash verification)
- Hard-link first / copy fallback when sending pool → version, so disk impact is usually zero on same-NTFS-volume

### Locked: strict no-fallback OCR
- `check_tracking.check_tracking(strict_no_box_fallback=True)` is the production default via pipeline_runner
- OCR runs ONLY on detected box crops. No center-region fallback, no full-frame fallback
- Reason: barcode-area crops produce garbage text that contaminated training and confused operators

### Locked: local-only, no DB, no cloud
- Pro: simple, deployable, no auth/secrets concerns
- Con: doesn't scale past one operator (workstation)
- Migration path: see Scaling section in README

### Locked: YOLOv8n (nano) for both tasks
- Pro: ~6 MB each, ~10 ms/frame on CPU, fits Electron bundle
- Con: ceiling on accuracy for tricky damage types
- When to upgrade: only after exhausting label-quality wins (see iteration loop)

### Locked: Streamlit as the operator UI
- Pro: zero deployment, runs over browser, fast iteration
- Con: rerun-everything model means certain UX patterns (modal dialogs, in-place edits) are hard. Documented footguns in memory under `reference_streamlit_checkbox_bulkops.md`
- Eventual migration: `serve.py` JSON-RPC + Electron front-end

### Critical QC rules vs warnings
- Critical (`too_short`, `no_motion`, `no_signal`, `blurry`) → status FAIL
- Warning (`too_dark`, `occluded`, `bad_framing`) → still listed but PASS

This is calibrated for handheld phone footage. Tune thresholds via CLI flags
when source quality changes (e.g. fixed CCTV).

---

## 7. Integration points

### 7.1 pipeline_runner — programmatic API

```python
from pipeline_runner import run_pipeline
result = run_pipeline(
    video_path="path/to.mp4",
    tracking_weights=None,  # None → auto-resolves latest models/rl_shipping_label_*
)
# result["final_status"] in {"AUTO_PASS", "REVIEW_REQUIRED", "NO_DAMAGE", "INVALID_INPUT"}
# result["tracking"]["detection"] has per-frame confidences + best bbox
# result["damage"]["detected"] / result["qc"]["status"]
```

The Streamlit pipeline tab calls this directly. An Electron front-end could spawn the same function via a Python sidecar.

### 7.2 Embedding the model in Electron (single-model legacy path)

Pattern: long-lived Python sidecar, newline-delimited JSON over stdin/stdout. `serve.py` covers this for the damage model.

```js
const py = spawn('python', ['scripts/serve.py',
  '--weights', 'models/rl_yolov8n/weights/best.pt', '--device', 'cpu'],
  { cwd: appResourcesPath });

const rl = readline.createInterface({ input: py.stdout });
rl.on('line', (line) => {
  const msg = JSON.parse(line);
  if (msg.event === 'ready') return;
  pending.get(msg.id)?.(msg);
});
```

Bundle Python via PyInstaller: `pyinstaller --onefile scripts/serve.py`,
ship in `extraResources`, no system Python required on user machine.

For the multi-model production shape, expose `pipeline_runner.run_pipeline` over the same JSON-RPC layer instead.

### 7.3 ONNX export

`train.py` exports `best.onnx` automatically. Use cases:
- Cross-platform inference (no Python runtime on target)
- Convert to TensorRT FP16 for ~2-3× throughput on NVIDIA
- Inspect graph at netron.app for debugging

### 7.4 Active learning hook

Two feedback channels feed back into the labeling loop:

1. **Review-tab CSV**: `outputs/review/review_<task>_<version>.csv` — operator-marked correct/incorrect on labels. Engineer copies ❌ frames into a new clean version (Train tab → 🧹 Clean dataset can do this with the `Also require is_correct=yes` checkbox).
2. **Auto-OCR corrections**: `<frame>.ocr.json` `corrected_text` fields — implicit ground-truth for tracking codes. Future fine-tuning could use the corrected/auto pairs as supervised signal.

---

## 8. Performance levers

On Intel i5-1335U (CPU only):

| Operation | Per video (~60s, 1800 frames) |
|---|---|
| OpenCV read + sample 10 frames | ~0.5 s |
| Brightness + Laplacian per frame | <0.1 s total |
| Frame-to-frame motion diff | <0.1 s |
| YOLOv8n inference on 12 frames | ~1.2 s |
| easyocr cold-start (first call) | 3–5 s — cached via `st.cache_resource` |
| easyocr per cropped label box | ~0.3 s |
| Multi-rotation OCR (4× rotations) | ~1.2 s per box (cached in sidecar) |
| **`pipeline_runner.run_pipeline`** | **~5–10 s per video** (cold cache) |

Notes:
- `qc_video.py` alone gates ~30 videos/min on a single CPU
- `pipeline_runner` overhead dominated by easyocr first call; subsequent runs same session re-use the cached Reader
- Auto-OCR in the Inspect tab runs synchronously, capped at 5 frames per render to keep the UI responsive

---

## 9. What this MVP intentionally lacks

- **No tracking** across frames in damage detection — every frame is independent. Adding a tracker (ByteTrack, BoT-SORT) would deduplicate the same physical damage across consecutive frames.
- **No per-class confidence calibration** — single `--conf` threshold per task. Real-world rollout will need per-class thresholds.
- **No drift monitoring** — production needs `confidence histogram per class per week → alert if mean drops > 5%`.
- **No real auth on multi-user** — `Current user` selectbox is a simulation. Add SSO + audit log before deploying to a shared workstation.
- **No file lock on assignments JSON** — atomic writes only. Two Streamlit instances claiming simultaneously could race; haven't observed in practice. Mitigation: `filelock` package, ~5 min change.
- **No fine-tuning loop on OCR corrections** — `<frame>.ocr.json` `corrected_text` fields are stored as ground truth but not yet fed back into a model. They could be — pair with the source crop and you have a supervised set for fine-tuning.
- **No timestamp-based filenames** — re-extracting the same video at the same fps produces the SAME filenames; carry-forward of labels works. Different fps → different `_fNNNN` indices → carry-forward needs Safe-mode hash verification. Filename change to `_t<seconds>` would break the load-bearing `_f` parser, so deferred.

These are the obvious next investments once labeling coverage and model accuracy are good enough to be worth productionizing.

---

## 10. Streamlit-specific footguns (documented in memory)

- **Checkbox bulk actions**: writing to a "shadow" set isn't enough; you must also write to each checkbox's `session_state[key]`, otherwise the widget's persisted True state re-adds the entry on rerun. See `reference_streamlit_checkbox_bulkops.md`.
- **Widget mutation rule**: can't write to `session_state[widget_key]` AFTER the widget is instantiated this run. Pattern: stage the value in `_pending_*::{...}` non-widget keys, apply at top of script before the widget is created.
- **streamlit-drawable-canvas Streamlit 1.30+ compat**: `image_to_url` was relocated. Shim is monkey-patched at `review_app.py` module load. Library itself is unused after the in-browser-labeling experiment was abandoned (mode-switching constraints), kept in venv harmlessly.
- **PyQt5 LabelImg float-type crashes**: 4 patches at `libs/canvas.py:526,530-531` and `libs/shape.py:131` (int()-wrap on QPainter coords).

These survive a venv rebuild only if reapplied — `reference_*.md` memory entries cover the steps.
