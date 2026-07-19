"""HuggingFace Hub sync — pull dataset to local dir, push local dir back."""
from __future__ import annotations

import os
from pathlib import Path


HF_TOKEN = os.environ.get("HUGGING_FACE_ACCESS_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Viet081197/orc-fdcr-dataset")
# Local data root used when running on HF Spaces
HF_LOCAL_DATA = Path(os.environ.get("ORC_DATA_ROOT", "/tmp/orc_data"))


def _hub():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def pull_from_hub(local_dir: Path | None = None) -> Path:
    """Download dataset repo snapshot → local_dir. Returns the local path."""
    from huggingface_hub import snapshot_download
    target = local_dir or HF_LOCAL_DATA
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(target),
        token=HF_TOKEN or None,
        ignore_patterns=["*.git*", ".gitattributes"],
    )
    return target


def push_to_hub(local_dir: Path | None = None, commit_message: str = "sync from review app") -> str:
    """Upload local_dir → HF Dataset repo. Returns commit URL."""
    from huggingface_hub import upload_folder
    src = local_dir or HF_LOCAL_DATA
    result = upload_folder(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        folder_path=str(src),
        token=HF_TOKEN,
        commit_message=commit_message,
        ignore_patterns=["*.pyc", "__pycache__", "*.git*", "*.part"],
    )
    return result.commit_url if hasattr(result, "commit_url") else str(result)
