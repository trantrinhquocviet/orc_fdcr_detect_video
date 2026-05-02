# Operator runbook — return-inspection labeling & review

For ops reviewers. You don't need to know Python or YOLO.

You'll do these jobs:

1. **Triage**: walk through extracted frames, mark which are worth labeling.
2. **Label**: draw boxes (in LabelImg, launched from the dashboard).
3. **Review**: after the model runs, mark its predictions correct or incorrect.

All three happen in the same browser-based dashboard (Streamlit).

---

## 1. Daily setup (~30 sec)

Open a terminal in the project folder and start the dashboard:

```bash
cd "C:/Users/OP-LT-0496/Downloads/ORC detect video"
.venv/Scripts/streamlit.exe run scripts/review_app.py
```

Browser opens at `http://localhost:8501`. Leave the terminal window open
the whole time. Closing the terminal stops the tool.

To stop at end of day: click on the terminal and press `Ctrl + C`.

---

## 2. The dashboard at a glance

You'll see **5 tabs** at the top and a **sidebar** on the left.

### Sidebar (always visible) — set these first

| Sidebar control | What to pick |
|---|---|
| **Training task** | `Damage` (damaged_item / empty_box) **or** `Shipping Label` (tracking-code text). The engineer will tell you which one to work on |
| **Show archived versions** | Leave unchecked unless the engineer asks |
| **Dataset version** | The newest is on top. Pick what the engineer says, or leave the default (newest) |
| **Current user** | Pick your name (`user_a`, `user_b`, …). Only the frames assigned to you will be listed |

### Tabs — one per job

| Tab | Use it for |
|---|---|
| **Review labels** | Section 5 — yes/no on existing labels |
| **Import frames** | Engineer's job — leave alone |
| **Inspect & label** | Section 3 — your main labeling screen |
| **Train** | Engineer's job — leave alone |
| **Run pipeline** | Engineer's job — leave alone |

---

## 3. Labeling workflow (Inspect & label tab)

**When**: engineer says "label these frames" and tells you which **task** + **version** to use.

**Goal**: draw a tight bounding box on each assigned frame.

### Step-by-step

1. Sidebar → set **Training task** + **Dataset version** + **Current user** as the engineer specified.
2. Click the **Inspect & label** tab.
3. Look at the **🔥 ACTIVE VERSION** banner at the top. Confirm it matches what the engineer asked for.
4. **Filters** (left to right):
   - **Status** = `Unlabeled` (so you only see what's left to do)
   - **Video** = `(all)` or pick a specific video
   - **Assignment** = `My work` (only your assigned frames)
5. **Claim frames** if your queue is empty:
   - Set Assignment filter to `Unassigned (claim)`
   - Tick the `📋 Queue` checkbox under each frame you want to take
   - Click **`👤 Claim N unassigned in queue`**
   - Set Assignment filter back to `My work`
6. Click any 🟡 thumbnail to inspect it. The frame opens above the grid with a red box drawn (if any labels exist).
7. Click **`🚀 Open in LabelImg`** at the top of the tab. A native desktop window opens at the selected image.
8. **In LabelImg**: see Section 4 below.
9. After saving in LabelImg, switch back to the browser:
   - Tick **`Auto-refresh (5s)`** at the top — badges flip 🟡 → 🟢 within 5 seconds of saving
   - Or click **`🔄 Refresh now`** to update immediately

### Per-card flags (under each thumbnail)

| Control | What it does |
|---|---|
| Image button (e.g. `🟢👤 0023`) | Click to inspect this frame in detail |
| `📋 Queue` checkbox | Adds the frame to your **labeling queue**. Open in LabelImg uses the queue's first 🟡 frame |
| `✅ Use` checkbox | UNCHECK to **exclude** this frame from training (e.g. you labeled it but the box is bad). Files stay on disk; the trainer just skips it |

### Badges at a glance

| Badge | Meaning |
|---|---|
| 🟢 | Has a saved label |
| 🟡 | Unlabeled |
| 👤 | Assigned to you |
| 🆓 | Unassigned (anyone can claim) |
| 👥 | Assigned to someone else (don't touch) |
| 🚫 | Excluded from training |
| ✏ | Has a user-corrected OCR text (only on Shipping Label task) |

---

## 4. LabelImg — drawing boxes

After you click **🚀 Open in LabelImg**, a native desktop window appears.

### One-time setup

- Maximize the LabelImg window
- Click the format toggle on the left toolbar until it says **YOLO** (NOT PascalVOC)
- Menu **View** → tick **Auto Save Mode** if available

### Hotkeys

| Key | Action |
|---|---|
| `W` | Start drawing a box (then click-drag) |
| `Ctrl + S` | Save the labels for this frame |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl + scroll` | Zoom |
| `Ctrl + F` | Fit window |

### Bounding box rules — DAMAGE task

Box only the damaged region:

- ✅ Tears, holes, punctures
- ✅ Crushed/dented corners
- ✅ Liquid stains, leaks
- ✅ Shattered product visible through opening

- ❌ Do NOT include hands, scissors, the table, the floor
- ❌ Do NOT box the whole package — only the damage
- Multiple tears on one package → ONE box covering all of them
- Unsure? Press `D` to skip — don't guess

For `empty_box` class:
- Open container clearly empty (or only filler, no product)

### Bounding box rules — SHIPPING LABEL task ⚠ STRICT

```
✅ Box the alphanumeric tracking text only — e.g. "TTVN1064832858"
❌ NEVER box the barcode above it
❌ NEVER box the whole shipping-label paper
```

The dashboard runs OCR on whatever you box. If you box the barcode, you get
garbage text and the model learns the wrong pattern. **Tight box around the
readable letters/numbers only.**

### When to skip

If you'd hesitate to draw a box (motion blur, finger covering, glare on the
text) → press `D` to advance without saving. Hedged labels are worse than
none.

---

## 5. Review labels workflow

**When**: engineer says "review the labels in <version>."

**Goal**: thumbs-up or thumbs-down each existing label so we know which to keep.

### Steps

1. Sidebar → set **Training task** + **Dataset version** as the engineer says.
2. Click the **Review labels** tab.
3. **Show only images with labels** is checked by default — leave it.
4. **Filter by video**: pick one video at a time, or leave at `(all)`.
5. For each frame, click:

   | Button | When |
   |---|---|
   | ✅ **Correct** | Box is in the right place + correct class. Pixel-perfect not required |
   | ❌ **Incorrect** | Box is wrong location, wrong class, or shouldn't exist |
   | **Next ➡** | Skip without voting |

6. Use **⏭ Next video** to jump to the next group when current video is done.

A CSV is written automatically: `outputs/review/review_<task>_<version>.csv`.

---

## 6. Common problems

| Problem | Fix |
|---|---|
| Streamlit page is blank or "Connection refused" | Terminal that started Streamlit was closed. Restart with the command in section 1 |
| LabelImg saves `.xml` files instead of `.txt` | Toggle the format button on the left toolbar to **YOLO** |
| LabelImg crashes when I draw a box | Already patched in this venv. If it happens again, take a screenshot of the error and ping the engineer |
| 🚀 Open in LabelImg button does nothing | Make sure the desktop is unlocked. Click any thumbnail FIRST so a frame is selected, THEN click the button |
| New labels not showing in dashboard after saving | Tick **Auto-refresh (5s)** at the top of the Inspect tab, or click **🔄 Refresh now** |
| I clicked the wrong button on a frame | Just keep going — the engineer can de-dupe. Most recent vote per image wins |
| Image is too small to see | In LabelImg: hold `Ctrl` + scroll. In dashboard: just look — square thumbnails are designed to fit the grid |
| I don't know if it's damaged / what to box | Skip it (`D` in LabelImg, or just don't draw). Hedged labels hurt the model |
| Sidebar version dropdown is missing my version | Tick **Show archived versions** in sidebar — it might be archived |
| My queue is empty | Set Assignment filter to `Unassigned (claim)`, tick frames, click **👤 Claim** |

---

## 7. What NOT to do

- Do not edit any file in `dataset/`, `models/`, `configs/` directly.
- Do not move or rename frames anywhere.
- Do not click anything in the **Train** or **Import frames** tab unless the engineer walks you through it.
- Do not install software.
- Do not commit changes to git unless the engineer asks.

If anything looks wrong or stuck, take a screenshot of the **whole browser tab** + the **terminal window** and ping the engineer.
