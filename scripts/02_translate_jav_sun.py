"""Augmentasi bahasa Jawa & Sunda: terjemahkan utterance MASSIVE id -> jav/sun dengan NLLB-200.

Label (intent) dipertahankan identik - hanya teks yang diterjemahkan. Output:
data/translated/massive-{jv,sun}.jsonl  dengan field {id, utt, intent, scenario, split, src}.

Catatan kualitas: ini data sintetis (MT). Terjemahan NLLB pada utterance pendek perintah
cukup baik, namun tetap ada noise - itulah sebabnya split test asli (id/en) tetap menjadi
metrik utama, dan hasil jv/su dibaca sebagai indikator adaptasi bahasa, bukan ground truth
manusia. Split train di-subsample (default 6000) untuk menjaga waktu GPU.
"""
import argparse
import json
import os
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "translated")

NLLB = "facebook/nllb-200-distilled-600M"
NLLB_TGT = {"jv": "jav_Latn", "sun": "sun_Latn", "su": "sun_Latn"}


def load_massive(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(RAW, "massive", "id-ID.jsonl"))
    p.add_argument("--langs", nargs="+", default=["jv", "sun"])
    p.add_argument("--max-train", type=int, default=6000)
    p.add_argument("--max-dev", type=int, default=1000)
    p.add_argument("--max-test", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=96)
    args = p.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from hflocal import local_snapshot

    os.makedirs(OUT, exist_ok=True)
    rows = load_massive(args.src)
    by_split = {"train": [], "dev": [], "test": []}
    for r in rows:
        by_split[r["partition"]].append(r)

    caps = {"train": args.max_train, "dev": args.max_dev, "test": args.max_test}
    rng = random.Random(42)
    selected = []
    for split, cap in caps.items():
        pool = by_split[split][:]
        rng.shuffle(pool)
        selected.extend(pool[:cap])
    print(f"total utterance diterjemahkan per bahasa: {len(selected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nllb_dir = local_snapshot(NLLB, ROOT)
    tok = AutoTokenizer.from_pretrained(nllb_dir, src_lang="ind_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_dir).to(device).eval()
    if device == "cuda":
        model = model.half()

    for code in args.langs:
        out_path = os.path.join(OUT, f"massive-{code}.jsonl")
        if os.path.exists(out_path):
            print(f"[{code}] sudah ada: {out_path} (hapus untuk ulang)")
            continue
        tgt = NLLB_TGT[code]
        bos_id = tok.convert_tokens_to_ids(tgt)
        results = []
        n = len(selected)
        for i in range(0, n, args.batch_size):
            chunk = selected[i:i + args.batch_size]
            texts = [r["utt"] for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos_id, num_beams=4,
                                     max_length=args.max_length)
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            for r, t in zip(chunk, outs):
                results.append({"id": r["id"], "utt": t.strip(), "intent": r["intent"],
                                "scenario": r["scenario"], "split": r["partition"], "src": "nllb-id"})
            done = min(i + args.batch_size, n)
            if done % (args.batch_size * 20) == 0 or done == n:
                print(f"[{code}] {done}/{n}")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{code}] tersimpan: {out_path}")


if __name__ == "__main__":
    main()
