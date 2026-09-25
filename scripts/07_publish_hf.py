"""Publish model + model card ke Hugging Face Hub.

Prasyarat: sudah `hf auth login` (atau set env HF_TOKEN).
Model card default bahasa Inggris (audiens internasional); `--lang id` untuk Indonesia.

Penggunaan:
  python scripts/07_publish_hf.py --repo-id <username-anda>/laya-idjvsuen-v1
  python scripts/07_publish_hf.py --repo-id ... --private   # repo privat dulu
  python scripts/07_publish_hf.py --repo-id ... --lang id   # card berbahasa Indonesia
"""
import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARDS = {"en": os.path.join(ROOT, "docs", "MODEL-CARD.en.md"),
         "id": os.path.join(ROOT, "docs", "MODEL-CARD.md")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="mis. username/laya-idjvsuen-v1")
    p.add_argument("--model-dir", default=os.path.join(ROOT, "runs", "laya-idjvsuen-v1"))
    p.add_argument("--lang", choices=["en", "id"], default="en",
                   help="bahasa model card yang diunggah sebagai README repo (default: en)")
    p.add_argument("--private", action="store_true", help="buat repo privat (bisa diubah nanti)")
    args = p.parse_args()

    from huggingface_hub import HfApi, get_token

    if not (get_token() or os.environ.get("HF_TOKEN")):
        sys.exit("Tidak ada token HF. Jalankan `hf auth login` dulu, atau set HF_TOKEN.")

    if not os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
        sys.exit(f"model.safetensors tidak ditemukan di {args.model_dir}.")

    api = HfApi()
    url = api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo siap: {url}")

    print("mengunggah bobot + tokenizer + config (±650 MB) ...")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint_latest*", ".cache*", "*.lock"],
    )

    # model card -> README.md repo, dengan placeholder repo-id diganti otomatis
    card = open(CARDS[args.lang], encoding="utf-8").read()
    namespace = args.repo_id.split("/")[0]
    card = (card
            .replace("<repo-id-ini>", args.repo_id)
            .replace("<this-repo-id>", args.repo_id)
            .replace("<user>/laya-idjvsuen", f"{namespace}/laya-idjvsuen"))
    if "<nama-anda>" in card or "<your-name>" in card:
        print("CATATAN: isi placeholder <nama-anda>/<your-name> di bagian sitasi "
              f"({CARDS[args.lang]}) lalu jalankan ulang skrip ini bila ingin nama Anda tercantum.")
    fd, tmp = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(card)
    api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    os.unlink(tmp)

    print(f"\nselesai: https://huggingface.co/{args.repo_id}")
    print("setelah review, buat publik di Settings repo bila awalnya --private.")


if __name__ == "__main__":
    main()
