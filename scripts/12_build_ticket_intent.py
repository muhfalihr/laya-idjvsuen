"""Bangun dataset ticket-intent (12 kategori domain) untuk fine-tune lanjutan v2.

- Split train/test per ticket_id (hash) agar pesan dari ticket sama tidak bocor
  lintas split.
- TRAIN: cap kategori dominan (service) untuk keseimbangan + replay data v1
  (NusaX sentimen + sampel MASSIVE) agar kemampuan lama tidak dilupakan.
- TEST: distribusi asli (mencerminkan realitas), format test_sets.pt (05_eval.py).

Output:
  data/processed/train_items_v2.pt        (item ticket + replay v1)
  data/processed/test_sets_tickets.pt     ({'ticket_intent': {...}} format 05_eval)
  data/processed/question_defs_v2.json
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
MODEL_V1 = os.path.join(ROOT, "runs", "laya-idjvsuen-v1")

MAX_LEN = 768
HEAD_MAX_LEN = 512
INSTRUCTIONS = "Classify the user's support request into the most fitting category."


def ticket_split(tid, train_ratio=0.8):
    h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
    return "train" if (h % 100) < (train_ratio * 100) else "test"


def build_item(tok, state, crit, gold_idx, n_opts, meta):
    from laya.common import build_sequence, render_options, QTYPES

    target = [0.0] * n_opts
    target[gold_idx] = 1.0
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": crit}
    k = len(render_options(q))
    seq, markers = build_sequence(tok, state, q, MAX_LEN, HEAD_MAX_LEN)
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES["choice"],
            "target": target, "label": gold_idx, "state": state, "meta": meta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_V1, help="checkpoint sumber tokenizer/config")
    p.add_argument("--labeled", default=os.path.join(PROC, "tickets_labeled.jsonl"))
    p.add_argument("--replay", default=os.path.join(PROC, "train_items.pt"))
    p.add_argument("--service-cap", type=int, default=1200)
    p.add_argument("--replay-massive", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config

    model_dir = args.model if os.path.exists(args.model) else None
    if model_dir is None:
        from hflocal import local_snapshot
        model_dir = local_snapshot("convaiinnovations/laya-multilingual", ROOT)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    labels = json.load(open(os.path.join(PROC, "tickets_label_dist.json"), encoding="utf-8"))["labels"]
    label_names = sorted(labels)  # urutan kanonik: alfabetis, identik train & eval
    crit = {name: labels[name] for name in label_names}
    idx = {n: i for i, n in enumerate(label_names)}
    print(f"{len(label_names)} kategori: {label_names}")

    rows = [json.loads(l) for l in open(args.labeled, encoding="utf-8")]
    train_rows = [r for r in rows if ticket_split(r["ticket_id"]) == "train"]
    test_rows = [r for r in rows if ticket_split(r["ticket_id"]) == "test"]
    print(f"tickets split: train {len(train_rows)} pesan / test {len(test_rows)} pesan")

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    service_count, train_capped = 0, []
    for r in train_rows:
        if r["label"] == "service":
            if service_count >= args.service_cap:
                continue
            service_count += 1
        train_capped.append(r)

    def to_items(rs):
        out = []
        for r in rs:
            it = build_item(tok, r["text"], crit, idx[r["label"]], len(label_names),
                            {"task": "ticket_intent", "lang": r["lang_guess"],
                             "gold": r["label"], "source": r["source"]})
            if it:
                out.append(it)
        return out

    ticket_items = to_items(train_capped)
    test_items = to_items(test_rows)
    print(f"items ticket: train {len(ticket_items)} (service di-cap {args.service_cap}) "
          f"/ test {len(test_items)}")
    print("distribusi train:", dict(Counter(i['meta']['gold'] for i in ticket_items)))
    print("distribusi test :", dict(Counter(i['meta']['gold'] for i in test_items)))

    # replay data v1: seluruh NusaX sentimen + sampel MASSIVE
    replay_items = []
    if os.path.exists(args.replay):
        old = torch.load(args.replay, weights_only=False)
        senti = [it for it in old if it.get("meta", {}).get("task") == "sentiment"]
        massive = [it for it in old if it.get("meta", {}).get("task") == "intent"]
        rng.shuffle(massive)
        replay_items = senti + massive[: args.replay_massive]
        print(f"replay v1: {len(senti)} sentimen + {min(len(massive), args.replay_massive)} MASSIVE")

    train_all = ticket_items + replay_items
    rng.shuffle(train_all)
    torch.save(train_all, os.path.join(PROC, "train_items_v2.pt"))
    print(f"TOTAL train v2: {len(train_all)}")

    torch.save({"ticket_intent": {"task": "ticket_intent", "lang": "mix",
                                  "items": test_items, "label_names": label_names,
                                  "instructions": INSTRUCTIONS, "criteria": crit}},
               os.path.join(PROC, "test_sets_tickets.pt"))
    with open(os.path.join(PROC, "question_defs_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"seq_cfg": {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN},
                   "ticket_intent": {"instructions": INSTRUCTIONS, "criteria": crit}},
                  f, ensure_ascii=False, indent=2)
    print("tersimpan: train_items_v2.pt, test_sets_tickets.pt, question_defs_v2.json")


if __name__ == "__main__":
    main()
