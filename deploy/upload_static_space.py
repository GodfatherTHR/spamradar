from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/spamradar"
    folder = Path(__file__).resolve().parent / "static_space"
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    print(f"repo: {url}", flush=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy free SpamRadar browser demo",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )
    print(f"commit: {commit.commit_url}", flush=True)
    info = api.space_info(repo_id)
    print(f"sdk: {getattr(info, 'sdk', None)}", flush=True)
    print(f"stage: {getattr(getattr(info, 'runtime', None), 'stage', None)}", flush=True)
    print(f"space: https://huggingface.co/spaces/{repo_id}", flush=True)
    print(f"app: https://{user}-spamradar.hf.space", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
