"""Bangun dataset GENERAL (jalur v4) dari sumber multibahasa + replay data v1.

Berbeda dengan tahap tiket (skrip 11-12), jalur ini memperluas kemampuan UMUM model
(bukan khusus ticketing) sambil menjaga bahasa id/jv/su/en lewat replay.

Sumber:
- SIB-200  : klasifikasi topik, paralel id/jav/sun/eng (FLORES-200; CC-BY)
- IndoNLU  : emosi (emot), sentimen app-review (smsa), aspek (casa) - opsional,
             dilewati lembut bila loader lama tidak kompatibel (butuh datasets<3.0)
- CS       : code-switch sintetis dari pasangan SIB-200 se-kategori (eval saja)
- Replay   : data v1 (train_items.pt) agar MASSIVE/NusaX/CS tidak dilupakan

Item identik skema skrip 03: {ids, markers, qtype, target, label, state, meta}.

Output (data/processed/):
- train_items_general.pt  : item train gabungan (shuffle)
- test_sets_general.pt    : set evaluasi BARU saja
- test_sets_v4.pt         : set lama (test_sets.pt) + baru, untuk 05_eval satu jalan
- question_defs_general.json

Penggunaan:
  python scripts/15_build_general.py --replay data/processed/train_items.pt
  python scripts/15_build_general.py --replay "" --cs-eval 0   # tanpa replay/CS
"""
import argparse
import json
import os
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")

# Samakan dengan skrip training (04_train.py) dan skrip 03.
MAX_LEN = 768
HEAD_MAX_LEN = 512

TOPIC_INSTRUCTIONS = "Classify the main topic of the text into the most likely category."
EMOT_INSTRUCTIONS = "What is the emotion expressed in the text?"
SMSA_INSTRUCTIONS = "What is the sentiment of the review?"
CASA_INSTRUCTIONS = "What is the sentiment towards the aspect mentioned in the review?"

# Deskripsi kriteria untuk 7 kategori kanonik SIB-200; kategori tak dikenal pakai fallback.
TOPIC_CRIT_DESC = {
    "geography": "about places, locations, geography, or the natural environment",
    "health": "about health, medicine, disease, or the human body",
    "politics": "about politics, government, elections, or public policy",
    "science/technology": "about science, technology, computers, or engineering",
    "sports": "about sports, athletes, games, or competitions",
    "travel": "about travel, tourism, destinations, or trips",
    "entertainment": "about entertainment, movies, music, or celebrities",
}
EMOT_CRIT_DESC = {
    "anger": "the text expresses anger or annoyance",
    "fear": "the text expresses fear or worry",
    "happy": "the text expresses happiness or joy",
    "happiness": "the text expresses happiness or joy",
    "love": "the text expresses love or affection",
    "sadness": "the text expresses sadness",
}
SENTI_CRIT = {
    "negative": "the text expresses a negative opinion, complaint, or dissatisfaction",
    "neutral": "the text is neutral, factual, or mixed",
    "positive": "the text expresses a positive opinion, praise, or satisfaction",
}

# Kode kandidat per bahasa: konfigurasi SIB-200 memakai kode FLORES-200 (varian
# antar repo allenai/Davlan), jadi coba beberapa sekaligus.
SIB_LANGS = {
    "id": ("ind_Latn", "ind", "id"),
    "jv": ("jav_Latn", "jav"),
    "sun": ("sun_Latn", "sun"),
    "en": ("eng_Latn", "eng", "en"),
}


def build_item(tok, seq_cfg, state, qtype_text, crit, gold_idx, n_opts, meta):
    """Sama dengan build_item skrip 03 (tipe choice)."""
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


def load_sib(lang):
    """Muat SIB-200 satu bahasa (repo Davlan, kode FLORES-200; terverifikasi 2026-09)."""
    from datasets import load_dataset

    for cfg in SIB_LANGS[lang]:
        try:
            ds = load_dataset("Davlan/sib200", cfg)
            return "Davlan/sib200", cfg, ds
        except Exception:
            continue
    return None, None, None


def sib_rows(ds, split):
    out = []
    for row in ds[split]:
        text = row.get("text") or row.get("sentence")
        cat = row.get("category") or row.get("label")
        if isinstance(cat, int):  # ClassLabel -> nama string
            feat = ds["train"].features.get("category") or ds["train"].features.get("label")
            cat = feat.names[cat] if feat else str(cat)
        if text and cat:
            out.append((text.strip(), str(cat).strip()))
    return out


def pick_col(row, names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def try_indonlu(cfg, task):
    """Muat satu config IndoNLU; kembalikan (train, test, label_names) atau None bila gagal.

    IndoNLU masih dataset skrip lama: datasets>=3 menolaknya; di datasets 2.x butuh
    trust_remote_code. Coba dua-duanya, gagal -> lewati lembut.
    """
    from datasets import load_dataset

    ds = None
    for kwargs in ({"trust_remote_code": True}, {}):
        try:
            ds = load_dataset("indonlp/indonlu", cfg, **kwargs)
            break
        except Exception:
            continue
    if ds is None:
        print(f"[{task}] dilewati (config '{cfg}' tidak bisa dimuat di versi datasets ini; "
              f"coba `pip install \"datasets<3.0\"`)")
        return None

    def norm_split(split):
        rows = []
        for row in ds[split]:
            text = pick_col(row, ("text", "sentence", "tweet", "review"))
            lab = pick_col(row, ("label", "sentiment", "emotion", "fine_grained_label",
                                 "label_1", "category"))
            if isinstance(lab, int):
                for feat in ds[split].features.values():
                    if hasattr(feat, "names") and len(feat.names) > lab:
                        lab = feat.names[lab]
                        break
            if not text or lab is None:
                continue
            extra = pick_col(row, ("aspect", "aspect_term", "target"))
            rows.append((str(text).strip(), str(lab).strip().lower(), extra))
        return rows

    tr, te = norm_split("train"), norm_split("test")
    labels = sorted({lab for _, lab, _ in tr} | {lab for _, lab, _ in te})
    if len(labels) < 2:
        print(f"[{task}] dilewati (label tidak terbaca)")
        return None
    return tr, te, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convaiinnovations/laya-multilingual",
                   help="direktori/HF-id untuk tokenizer+config (tokenizer sama dengan v1)")
    p.add_argument("--replay", default=os.path.join(PROCESSED, "train_items.pt"),
                   help="path data replay v1; kosongkan ('') untuk tanpa replay")
    p.add_argument("--replay-frac", type=float, default=1.0,
                   help="proporsi data replay yang dipakai (0-1)")
    p.add_argument("--max-sib-train-per-lang", type=int, default=0, help="0 = semua")
    p.add_argument("--max-sib-test-per-lang", type=int, default=400)
    p.add_argument("--cs-eval", type=int, default=300,
                   help="jumlah item code-switch sintetis per kombinasi (0 = nonaktif)")
    p.add_argument("--indonlu", action="store_true", default=True,
                   help="sertakan subset IndoNLU (emot/smsa/casa) bila bisa dimuat")
    p.add_argument("--no-indonlu", dest="indonlu", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from hflocal import local_snapshot

    model_dir = args.model if os.path.isdir(args.model) else local_snapshot(args.model, ROOT)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    seq_cfg = {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}

    os.makedirs(PROCESSED, exist_ok=True)
    rng = random.Random(args.seed)
    train_items, test_sets = [], {}

    # ---- 1) SIB-200: topik, 4 bahasa ----
    sib_data = {}
    for lang in SIB_LANGS:
        repo, cfg, ds = load_sib(lang)
        if ds is None:
            print(f"[sib/{lang}] GAGAL dimuat - lewati bahasa ini "
                  f"(cek koneksi/ketersediaan {list(SIB_LANGS[lang])})")
            continue
        tr = sib_rows(ds, "train") + sib_rows(ds, "validation")  # dev ikut train
        te = sib_rows(ds, "test")
        if args.max_sib_train_per_lang and len(tr) > args.max_sib_train_per_lang:
            rng.shuffle(tr)
            tr = tr[: args.max_sib_train_per_lang]
        if len(te) > args.max_sib_test_per_lang:
            te = rng.sample(te, args.max_sib_test_per_lang)
        sib_data[lang] = {"train": tr, "test": te}
        print(f"[sib/{lang}] {repo}/{cfg}: train {len(tr)}, test {len(te)}")

    topics = sorted({cat for d in sib_data.values() for _, cat in d["train"] + d["test"]})
    if topics:
        topic_idx = {t: i for i, t in enumerate(topics)}
        topic_crit = {t: TOPIC_CRIT_DESC.get(t, f"about {t}") for t in topics}
        print(f"[sib] {len(topics)} kategori topik: {topics}")

        for lang, d in sib_data.items():
            items = []
            for text, cat in d["train"]:
                if cat not in topic_idx:
                    continue
                meta = {"lang": lang, "task": "topic", "gold": cat,
                        "instructions": TOPIC_INSTRUCTIONS, "source": "sib200"}
                it = build_item(tok, seq_cfg, text, "choice", topic_crit,
                                topic_idx[cat], len(topics), meta)
                if it:
                    items.append(it)
            train_items.extend(items)
            print(f"[topic/{lang}] train: {len(items)}")

            items = []
            for text, cat in d["test"]:
                if cat not in topic_idx:
                    continue
                meta = {"lang": lang, "task": "topic", "gold": cat,
                        "instructions": TOPIC_INSTRUCTIONS, "source": "sib200"}
                it = build_item(tok, seq_cfg, text, "choice", topic_crit,
                                topic_idx[cat], len(topics), meta)
                if it:
                    items.append(it)
            test_sets[f"sib_{lang}"] = {
                "task": "topic", "lang": lang, "items": items,
                "label_names": topics, "instructions": TOPIC_INSTRUCTIONS,
                "criteria": topic_crit,
            }
            print(f"[topic/{lang}] test: {len(items)}")

    # ---- 2) IndoNLU (opsional): emot / smsa / casa ----
    if args.indonlu:
        def add_indonlu(task, cfg, instr, crit_base, state_fn):
            res = try_indonlu(cfg, task)
            if not res:
                return
            tr, te, labels = res
            crit = {lb: crit_base.get(lb, f"the text is {lb}") for lb in labels}
            idx = {lb: i for i, lb in enumerate(labels)}
            for split_rows, dest in ((tr, "train"), (te, "test")):
                items = []
                for text, lab, extra in split_rows:
                    if lab not in idx:
                        continue
                    state = state_fn(text, extra)
                    meta = {"lang": "id", "task": task, "gold": lab,
                            "instructions": instr, "source": "indonlu"}
                    it = build_item(tok, seq_cfg, state, "choice", crit,
                                    idx[lab], len(labels), meta)
                    if it:
                        items.append(it)
                if dest == "train":
                    train_items.extend(items)
                    print(f"[{task}/id] train: {len(items)}")
                elif items:
                    test_sets[f"{task}_id"] = {
                        "task": task, "lang": "id", "items": items,
                        "label_names": labels, "instructions": instr, "criteria": crit,
                    }
                    print(f"[{task}/id] test: {len(items)}")

        add_indonlu("emot", "emot", EMOT_INSTRUCTIONS, EMOT_CRIT_DESC,
                    lambda text, extra: text)
        add_indonlu("smsa", "smsa", SMSA_INSTRUCTIONS, SENTI_CRIT,
                    lambda text, extra: text)
        add_indonlu("casa", "casa", CASA_INSTRUCTIONS, SENTI_CRIT,
                    lambda text, extra: f"{text} — aspect: {extra}" if extra else text)

    # ---- 3) Code-switch sintetis dari SIB-200 (eval saja) ----
    if args.cs_eval > 0 and len(sib_data) >= 2:
        def half_mix(a_text, b_text):
            wa, wb = a_text.split(), b_text.split()
            return " ".join(wa[: (len(wa) + 1) // 2] + wb[len(wb) // 2:]).strip()

        combos = [("id", "en"), ("en", "id"), ("id", "jv"), ("jv", "id"),
                  ("id", "sun"), ("sun", "id")]
        topics_c = topics
        topic_crit_c = {t: TOPIC_CRIT_DESC.get(t, f"about {t}") for t in topics_c}
        for la, lb in combos:
            if la not in sib_data or lb not in sib_data:
                continue
            # pasangkan se-kategori agar gold tetap valid untuk kedua belahan
            by_cat_a, by_cat_b = {}, {}
            for text, cat in sib_data[la]["train"]:
                by_cat_a.setdefault(cat, []).append(text)
            for text, cat in sib_data[lb]["train"]:
                by_cat_b.setdefault(cat, []).append(text)
            items = []
            cats = sorted(set(by_cat_a) & set(by_cat_b))
            quota = {c: 0 for c in cats}
            pairs = []
            for c in cats:
                A, B = list(by_cat_a[c]), list(by_cat_b[c])
                rng.shuffle(A)
                rng.shuffle(B)
                pairs.extend((a, b, c) for a, b in zip(A, B))
            rng.shuffle(pairs)
            for a_text, b_text, cat in pairs[: args.cs_eval * 3]:
                if quota[cat] >= args.cs_eval:
                    continue
                state = half_mix(a_text, b_text)
                meta = {"lang": f"{la}+{lb}", "task": "topic", "gold": cat,
                        "instructions": TOPIC_INSTRUCTIONS, "source": "sib200",
                        "synthetic": True}
                it = build_item(tok, seq_cfg, state, "choice", topic_crit_c,
                                topics_c.index(cat), len(topics_c), meta)
                if it:
                    items.append(it)
                    quota[cat] += 1
            if items:
                test_sets[f"cs_sib_{la}_{lb}"] = {
                    "task": "topic", "lang": f"{la}+{lb}", "items": items,
                    "label_names": topics_c, "instructions": TOPIC_INSTRUCTIONS,
                    "criteria": topic_crit_c,
                }
                print(f"[cs_sib/{la}+{lb}] eval sintetis: {len(items)}")

    # ---- 4) Replay data v1 ----
    n_new = len(train_items)
    if args.replay and os.path.exists(args.replay):
        replay = torch.load(args.replay, weights_only=False)
        if args.replay_frac < 1.0:
            rng.shuffle(replay)
            replay = replay[: int(len(replay) * args.replay_frac)]
        train_items.extend(replay)
        print(f"[replay] {len(replay)} item dari {args.replay}")
    elif args.replay:
        print(f"[replay] {args.replay} tidak ditemukan - lanjut TANPA replay")

    # ---- simpan ----
    rng.shuffle(train_items)
    torch.save(train_items, os.path.join(PROCESSED, "train_items_general.pt"))
    torch.save(test_sets, os.path.join(PROCESSED, "test_sets_general.pt"))
    with open(os.path.join(PROCESSED, "question_defs_general.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seq_cfg": seq_cfg,
            "topic": {"instructions": TOPIC_INSTRUCTIONS,
                      "criteria": {t: TOPIC_CRIT_DESC.get(t, f"about {t}") for t in topics}},
        }, f, ensure_ascii=False, indent=2)

    old_sets_path = os.path.join(PROCESSED, "test_sets.pt")
    if os.path.exists(old_sets_path):
        merged = torch.load(old_sets_path, weights_only=False)
        merged.update(test_sets)
        torch.save(merged, os.path.join(PROCESSED, "test_sets_v4.pt"))
        print(f"merged: test_sets_v4.pt ({len(merged)} set)")

    print(f"\nTOTAL train items: {len(train_items)} "
          f"(item baru {n_new} + replay {len(train_items) - n_new})")
    print("tersimpan: train_items_general.pt, test_sets_general.pt, question_defs_general.json")


if __name__ == "__main__":
    main()
