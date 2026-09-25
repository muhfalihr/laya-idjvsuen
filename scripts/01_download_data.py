"""Download data mentah: MASSIVE 1.1 (id, en) + NusaX-senti (id, jav, sun, en).

Sumber primer:
- MASSIVE 1.1 (CC BY 4.0, Amazon): https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
- NusaX-senti (CC-BY-SA 4.0, IndoNLP): https://github.com/IndoNLP/nusax  datasets/sentiment/<lang>/

Output: data/raw/massive/{id-ID,en-US}.jsonl, data/raw/nusax/{<code>}/{train,valid,test}.csv
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_FILES = {"id-ID.jsonl", "en-US.jsonl"}

NUSAX_LANGS = {"id": "indonesian", "en": "english", "jv": "javanese", "sun": "sundanese"}
NUSAX_SPLITS = ("train", "valid", "test")
NUSAX_BASE = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment"


def download_massive():
    out_dir = os.path.join(RAW, "massive")
    os.makedirs(out_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(out_dir, f)) for f in MASSIVE_FILES):
        print("[massive] sudah ada, lewati download")
        return
    print("[massive] men-download tarball 1.1 (~200-300 MB) ...")
    req = urllib.request.Request(MASSIVE_URL, headers={"User-Agent": "laya-id-finetune"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    print(f"[massive] tarball {len(data)/1e6:.1f} MB diterima, ekstrak {sorted(MASSIVE_FILES)}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        wanted = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.name.split("/")[-1] in MASSIVE_FILES}
        missing = MASSIVE_FILES - set(wanted)
        if missing:
            sys.exit(f"[massive] file tidak ditemukan di tarball: {missing}")
        for name, member in wanted.items():
            f = tf.extractfile(member)
            with open(os.path.join(out_dir, name), "wb") as out:
                out.write(f.read())
            print(f"[massive] {name} tersimpan")


def download_nusax():
    for code, folder in NUSAX_LANGS.items():
        out_dir = os.path.join(RAW, "nusax", code)
        os.makedirs(out_dir, exist_ok=True)
        for split in NUSAX_SPLITS:
            dest = os.path.join(out_dir, f"{split}.csv")
            if os.path.exists(dest):
                continue
            url = f"{NUSAX_BASE}/{folder}/{split}.csv"
            print(f"[nusax] {code}/{split}.csv ...")
            req = urllib.request.Request(url, headers={"User-Agent": "laya-id-finetune"})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as out:
                out.write(r.read())


def stats():
    import json

    for lang_file in sorted(MASSIVE_FILES):
        path = os.path.join(RAW, "massive", lang_file)
        counts, intents = {}, set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                counts[row["partition"]] = counts.get(row["partition"], 0) + 1
                intents.add(row["intent"])
        print(f"[massive] {lang_file}: {counts} | intent unik: {len(intents)}")

    import csv

    for code in NUSAX_LANGS:
        line = []
        for split in NUSAX_SPLITS:
            with open(os.path.join(RAW, "nusax", code, f"{split}.csv"), encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
            line.append(f"{split}={n}")
        print(f"[nusax] {code}: {' '.join(line)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()
    if not args.stats_only:
        download_massive()
        download_nusax()
    stats()
