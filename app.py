"""HuggingFace Spaces entry point.

Sets ORC_DATA_ROOT to /tmp/orc_data, pulls dataset from HF Hub on first run,
then delegates to scripts/review_app.py.
"""
import os
import sys
from pathlib import Path

DATA_ROOT = Path("/tmp/orc_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ORC_DATA_ROOT", str(DATA_ROOT))

# Pull dataset on cold start (only if hub token present and data not yet downloaded)
_marker = DATA_ROOT / ".pulled"
if not _marker.exists() and os.environ.get("HUGGING_FACE_ACCESS_TOKEN"):
    try:
        from hf_sync import pull_from_hub
        pull_from_hub(DATA_ROOT)
        _marker.touch()
    except Exception as e:
        print(f"[hf_sync] pull failed (non-fatal): {e}", file=sys.stderr)

# Add scripts/ to path so review_app can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from scripts.review_app import main  # noqa: E402
main()
