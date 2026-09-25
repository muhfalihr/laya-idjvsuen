"""Helper download model HF tanpa symlink (Windows non-admin).

hf-cache default memakai symlink ke blob store, yang di Windows butuh Developer
Mode/admin. Mode local_dir menulis file nyata, jadi aman untuk semua pengguna.
"""
import os

from huggingface_hub import snapshot_download

ALLOW = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*",
         "*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "*.model"]


def local_snapshot(model_id: str, root: str) -> str:
    dest = os.path.join(root, "models", model_id.replace("/", "__"))
    return snapshot_download(model_id, local_dir=dest, allow_patterns=ALLOW)
