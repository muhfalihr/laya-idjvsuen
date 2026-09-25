"""Bangun dataset training + evaluasi dalam format typed-decisions Laya (choice).

Tugas & bahasa:
- intent   : MASSIVE 60-way, bahasa id, en (asli) + jv, sun (terjemahan NLLB bila ada)
- sentiment: NusaX-senti 3-way, bahasa id, jv, sun, en

Item yang dihasilkan identik skemanya dengan notebook resmi
(laya_finetune_typed_decisions_2xT4_kaggle.ipynb): {ids, markers, qtype, target, label}
plus meta {lang, task, gold} untuk analisis.

Override token budget (relatif config checkpoint multilingual max_len=1024/head_max_len=256):
head_max_len=512 agar 60 opsi intent tidak terpotong menjadi awalan 3 token yang sama,
max_len=768 untuk margin state panjang. Nilai ini ditulis ke config checkpoint hasil training
sehingga evaluasi via Agent memakai budget yang sama.
"""
import argparse
import csv
import json
import os
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
TRANSLATED = os.path.join(ROOT, "data", "translated")
PROCESSED = os.path.join(ROOT, "data", "processed")

# Samakan dengan skrip training (04_train.py).
MAX_LEN = 768
HEAD_MAX_LEN = 512

INTENT_INSTRUCTIONS = "Classify the user's utterance into the most likely intent."
SENTI_LABELS = ["negative", "neutral", "positive"]
SENTI_CRIT = {
    "negative": "the text expresses a negative opinion, complaint, or dissatisfaction",
    "neutral": "the text is neutral, factual, or mixed",
    "positive": "the text expresses a positive opinion, praise, or satisfaction",
}
SENTI_INSTRUCTIONS = "What is the sentiment of the text?"


def read_massive(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_nusax(code, split):
    path = os.path.join(RAW, "nusax", code, f"{split}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return [(row["text"], row["label"].strip().lower()) for row in csv.DictReader(f)]


def build_item(tok, seq_cfg, state, qtype_text, crit, gold_idx, n_opts, meta):
    """Replika logika build_training_item notebook untuk tipe choice."""
    from laya.common import build_sequence, render_options, QTYPES

    target = [0.0] * n_opts
    target[gold_idx] = 1.0
    q = {"t": qtype_text, "ins": meta["instructions"], "crit": crit}
    k = len(render_options(q))
    seq, markers = build_sequence(tok, state, q, seq_cfg["max_len"], seq_cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[qtype_text],
        "target": target,
        "label": gold_idx,
        "state": state,
        "meta": {kk: vv for kk, vv in meta.items() if kk != "instructions"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convaiinnovations/laya-multilingual")
    p.add_argument("--max-intent-train-per-lang", type=int, default=0,
                   help="0 = pakai semua (11.5k per bahasa)")
    p.add_argument("--max-intent-test-per-lang", type=int, default=2000)
    p.add_argument("--cs-eval", type=int, default=500,
                   help="jumlah pasangan code-switch sintetis per kombinasi (0 = nonaktif)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from hflocal import local_snapshot

    model_dir = local_snapshot(args.model, ROOT)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    seq_cfg = {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}

    os.makedirs(PROCESSED, exist_ok=True)

    # ---- intent: opsi kanonik (urutan identik untuk train & eval!) ----
    id_rows = read_massive(os.path.join(RAW, "massive", "id-ID.jsonl"))
    en_rows = read_massive(os.path.join(RAW, "massive", "en-US.jsonl"))
    intents = sorted({r["intent"] for r in id_rows} | {r["intent"] for r in en_rows})
    intent_idx = {name: i for i, name in enumerate(intents)}
    intent_crit = {name: None for name in intents}  # opsi = nama intent saja
    print(f"[intent] {len(intents)} kelas intent")
    assert len(intents) > 50, "jumlah intent MASSIVE tidak wajar"

    train_items, test_sets = [], {}
    rng = random.Random(args.seed)

    def add_intent_set(rows, lang, split_name, cap):
        items = []
        for r in rows:
            meta = {"lang": lang, "task": "intent", "gold": r["intent"],
                    "instructions": INTENT_INSTRUCTIONS}
            it = build_item(tok, seq_cfg, r["utt"], "choice", intent_crit,
                            intent_idx[r["intent"]], len(intents), meta)
            if it:
                items.append(it)
        if cap and len(items) > cap:
            items = rng.sample(items, cap)
        if split_name == "train":
            train_items.extend(items)
            print(f"[intent/{lang}] train: {len(items)}")
        else:
            test_sets[f"intent_{lang}"] = {
                "task": "intent", "lang": lang, "items": items, "label_names": intents,
                "instructions": INTENT_INSTRUCTIONS, "criteria": intent_crit,
            }
            print(f"[intent/{lang}] {split_name}: {len(items)}")

    for rows, lang in ((id_rows, "id"), (en_rows, "en")):
        tr = [r for r in rows if r["partition"] == "train"]
        te = [r for r in rows if r["partition"] == "test"]
        de = [r for r in rows if r["partition"] == "dev"]
        if args.max_intent_train_per_lang:
            rng.shuffle(tr)
            tr = tr[:args.max_intent_train_per_lang]
        add_intent_set(tr + de, lang, "train", 0)  # dev ikut train (kalibrasi di-hold-out di train script)
        add_intent_set(te, lang, "test", args.max_intent_test_per_lang)

    # jv/sun hasil terjemahan (opsional)
    for code, lang in (("jv", "jv"), ("sun", "sun")):
        path = os.path.join(TRANSLATED, f"massive-{code}.jsonl")
        if os.path.exists(path):
            rows = read_massive(path)
            tr = [r for r in rows if r["split"] in ("train", "dev")]
            te = [r for r in rows if r["split"] == "test"]
            add_intent_set(tr, lang, "train", 0)
            add_intent_set(te, lang, "test", args.max_intent_test_per_lang)
        else:
            print(f"[intent/{lang}] dilewati (belum ada {path}; jalankan 02_translate_jav_sun.py)")

    # ---- sentiment NusaX ----
    nusax_codes = {"id": "id", "jv": "jv", "sun": "sun", "en": "en"}
    for code, lang in nusax_codes.items():
        for split, split_name in (("train", "train"), ("test", "test")):
            rows = read_nusax(code, split)
            items = []
            for text, lab in rows:
                if lab not in SENTI_LABELS:
                    continue
                meta = {"lang": lang, "task": "sentiment", "gold": lab,
                        "instructions": SENTI_INSTRUCTIONS}
                it = build_item(tok, seq_cfg, text, "choice", SENTI_CRIT,
                                SENTI_LABELS.index(lab), len(SENTI_LABELS), meta)
                if it:
                    items.append(it)
            if split_name == "train":
                train_items.extend(items)
                print(f"[senti/{lang}] train: {len(items)}")
            else:
                test_sets[f"sentiment_{lang}"] = {
                    "task": "sentiment", "lang": lang, "items": items,
                    "label_names": SENTI_LABELS, "instructions": SENTI_INSTRUCTIONS,
                    "criteria": SENTI_CRIT,
                }
                print(f"[senti/{lang}] test: {len(items)}")

    # ---- code-switch sintetis (eval saja): belahan kalimat A + belahan kalimat B ----
    if args.cs_eval > 0:
        id_te = [r for r in id_rows if r["partition"] == "test"]
        en_te = [r for r in en_rows if r["partition"] == "test"]
        # MASSIVE paralel per id (intent identik antar bahasa)
        id2en = {r["id"]: r for r in en_te}
        combos = [("id", "en"), ("en", "id")]
        # pasangan id-jv / id-sun dari hasil terjemahan
        tr_maps = {}
        for code, lang in (("jv", "jv"), ("sun", "sun")):
            path = os.path.join(TRANSLATED, f"massive-{code}.jsonl")
            if os.path.exists(path):
                tr_maps[lang] = {r["id"]: r for r in read_massive(path) if r["split"] == "test"}
                combos.extend([("id", lang), (lang, "id")])

        def half_mix(a_text, b_text):
            wa, wb = a_text.split(), b_text.split()
            return " ".join(wa[: (len(wa) + 1) // 2] + wb[len(wb) // 2:]).strip()

        for la, lb in combos:
            items = []
            pool = [r for r in id_te if r["id"] in id2en]
            if lb != "en":
                pool = [r for r in pool if r["id"] in tr_maps.get(lb, {})]
            rng.shuffle(pool)
            for a in pool[: args.cs_eval]:
                a_text = a["utt"] if la == "id" else (id2en[a["id"]]["utt"] if la == "en" else tr_maps[la][a["id"]]["utt"])
                b_text = (id2en[a["id"]]["utt"] if lb == "en"
                          else (a["utt"] if lb == "id" else tr_maps[lb][a["id"]]["utt"]))
                state = half_mix(a_text, b_text)
                meta = {"lang": f"{la}+{lb}", "task": "intent", "gold": a["intent"],
                        "instructions": INTENT_INSTRUCTIONS, "synthetic": True}
                it = build_item(tok, seq_cfg, state, "choice", intent_crit,
                                intent_idx[a["intent"]], len(intents), meta)
                if it:
                    items.append(it)
            if items:
                test_sets[f"cs_{la}_{lb}"] = {
                    "task": "intent", "lang": f"{la}+{lb}", "items": items,
                    "label_names": intents, "instructions": INTENT_INSTRUCTIONS,
                    "criteria": intent_crit,
                }
                print(f"[cs/{la}+{lb}] eval sintetis: {len(items)}")

    # ---- simpan ----
    rng.shuffle(train_items)
    torch.save(train_items, os.path.join(PROCESSED, "train_items.pt"))
    print(f"\nTOTAL train items: {len(train_items)}")
    with open(os.path.join(PROCESSED, "question_defs.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seq_cfg": seq_cfg,
            "intent": {"instructions": INTENT_INSTRUCTIONS, "criteria": intent_crit},
            "sentiment": {"instructions": SENTI_INSTRUCTIONS, "criteria": SENTI_CRIT},
        }, f, ensure_ascii=False, indent=2)
    torch.save(test_sets, os.path.join(PROCESSED, "test_sets.pt"))
    print("tersimpan: train_items.pt, test_sets.pt, question_defs.json di data/processed/")


if __name__ == "__main__":
    main()
