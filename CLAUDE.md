# Claude project instructions

Read this before responding to anything in this repo.

---

## Project context

Local YOLOv8 pipeline for warehouse return-package inspection. Two classes
only (MVP): `damaged_item`, `empty_box`. Single operator, no cloud, runs on
Intel i5-1335U CPU. See `README.md` for the full architecture.

---

## Locked-in decisions — do not relitigate

1. **Two classes only** (`damaged_item`, `empty_box`). Do not propose a 3rd
   class unless the user explicitly asks. The original 5-class scheme in
   git history is deprecated for MVP.

2. **Strict video-stratified split.** `scripts/split.py` requires ≥2 video_ids
   and exits non-zero otherwise. Do not re-add the per-image fallback that
   existed briefly on 2026-04-30 — the user explicitly removed it because
   it leaks frames between train and val.

3. **Filename contract is load-bearing**: `<video_id>_fNNNN.jpg`.
   - `video_id`: alphanumeric + dashes only, no underscores
   - `_f` is the parser delimiter; never put `_f` inside a video_id
   - Several scripts (`split.py`, `review_app.py`) silently corrupt if this breaks

4. **Local tooling only**: LabelImg + Streamlit. No Roboflow, no CVAT, no
   cloud labeling, no databases.

5. **CPU training**: no CUDA on the user's machine. Default `--device cpu --batch 4`.

---

## Tooling gotchas worth remembering

- `labelImg.exe` from the venv fails to launch from Claude's bash on Windows
  (exit code 127). Use `python .venv/Lib/site-packages/labelImg/labelImg.py ...`
  instead. The user must launch GUI apps themselves — Claude can't reliably
  spawn them.
- Streamlit auto-reloads on file change. Tell the user to refresh the page,
  not to restart the server.
- `python -m labelImg` doesn't work — labelImg is a package without `__main__`.
- The `_raw/` directory contains both images and `.txt` labels side by side;
  keep them together.

---

## MVP iteration discipline

- **Speed > accuracy**. The user explicitly wants fast iteration on small datasets.
- Don't add features the task doesn't require. Don't refactor. Don't
  introduce abstractions for hypothetical future requirements.
- Don't write multi-paragraph docstrings. One short line max in code comments.
- Default to no comments unless the WHY is non-obvious.
- Don't propose 50-frame labeling sessions when 10 will do.

---

## When suggesting next steps

- Don't suggest retraining until: ≥40 labeled frames across ≥4 video_ids
  with both classes represented.
- After each training run, look at `models/<name>/results.png` and
  `val_batch0_pred.jpg` — pick the **single worst symptom** and fix only that.
- Stop tweaking the model when mAP@50 < 0.3 after 2 retrains; the labels
  are the problem.

---

## Behavioral signals to remember

- The user is the technical lead, not a beginner. Skip basic explanations.
- They prefer terse responses, tables, and copy-paste-ready commands.
- They've explicitly asked for execution-focused output ("Output format:
  Clear steps. Minimal theory. Focus on execution.").
- They use the IDE's open-file hint (`<system-reminder>` for IDE file open)
  as a soft signal — sometimes related, sometimes not. Don't auto-act on it.

---

## Memory system

The persistent memory at `C:\Users\OP-LT-0496\.claude\projects\C--Users-OP-LT-0496-Downloads-ORC-detect-video\memory\`
already contains:

- `MEMORY.md` (index)
- `project_state.md` (current state, paused decision points)
- `project_decisions.md` (the "locked decisions" above, with rationale)

Update these when project state changes (new training run, new tooling,
shifted decisions). Don't duplicate this CLAUDE.md content into memory — this
file is already in the repo.

---

## Output style

- English by default; Vietnamese if the user writes in Vietnamese.
- Tables for comparisons / status / parameters.
- Code fences for commands, with the explicit `cd` step or `.venv/Scripts/...`
  paths the user has been using.
- No emojis in files. Emojis OK in chat responses (the user has used them).
