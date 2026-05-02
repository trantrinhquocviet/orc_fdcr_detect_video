"""Rule engine: turns raw class detections into business-level issues.

Inputs come from `infer.py` as a per-frame list of detected class names.
Output is the high-level risk classification a returns operator cares about.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

# Classes the YOLO model emits.
CLASSES = ("box", "open_box", "damaged_item", "empty_box", "no_seal")

# Severity weights are used only for sorting / UI prioritization.
SEVERITY = {
    "high_risk": 3,
    "missing_item": 2,
    "damage": 2,
    "open_box": 1,
    "no_seal": 1,
    "ok": 0,
}


def classify_frame(class_names: Iterable[str]) -> list[str]:
    """Map raw YOLO classes on a single frame to business issues."""
    present = set(class_names)
    issues: list[str] = []

    # Ordering of rules matters: most severe first so UI can short-circuit.
    if "open_box" in present and "no_seal" in present:
        issues.append("high_risk")  # tampering / theft signal
    if "empty_box" in present:
        issues.append("missing_item")
    if "damaged_item" in present:
        issues.append("damage")
    if "open_box" in present and "high_risk" not in issues:
        issues.append("open_box")
    if "no_seal" in present and "high_risk" not in issues:
        issues.append("no_seal")

    return issues or ["ok"]


def aggregate(per_frame_issues: list[list[str]]) -> dict:
    """Combine per-frame issues into a file-level verdict."""
    flat: list[str] = [i for frame in per_frame_issues for i in frame if i != "ok"]
    counts = Counter(flat)
    unique = sorted(counts.keys(), key=lambda k: SEVERITY.get(k, 0), reverse=True)

    if not unique:
        verdict = "ok"
    elif "high_risk" in counts:
        verdict = "high_risk"
    elif "missing_item" in counts:
        verdict = "missing_item"
    elif "damage" in counts:
        verdict = "damage"
    else:
        verdict = unique[0]

    return {
        "verdict": verdict,
        "issues_detected": unique,
        "issue_counts": dict(counts),
    }
