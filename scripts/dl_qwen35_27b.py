import os
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
p = snapshot_download(
    repo_id="Qwen/Qwen3.5-27B",
    cache_dir=os.path.expanduser("~/JasonLucas/hf_cache/hub"),
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"],
    max_workers=4,
    resume_download=True,
)
print("DONE", p)
