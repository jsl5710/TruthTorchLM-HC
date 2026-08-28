#!/usr/bin/env python
"""Robust teacher-model downloader for the multi-target RQ. Retries through transient
HF stalls; a short HF_HUB_DOWNLOAD_TIMEOUT turns silent read-stalls into raised errors
so the loop resumes instead of hanging. Resumes from .incomplete blobs automatically.

    HF_HUB_DOWNLOAD_TIMEOUT=30 python scripts/dl_teacher.py Qwen/Qwen3.5-27B
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

repo = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2  # low workers -> stay under login-node mem cap
cache = os.path.expanduser("~/JasonLucas/hf_cache/hub")

for attempt in range(1, 500):
    try:
        p = snapshot_download(
            repo_id=repo,
            cache_dir=cache,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
            max_workers=workers,
        )
        print("DONE", repo, p, flush=True)
        break
    except Exception as e:  # noqa: BLE001 -- resume through any transient network error
        print(f"[attempt {attempt}] retry after: {str(e)[:200]}", flush=True)
        time.sleep(10)
