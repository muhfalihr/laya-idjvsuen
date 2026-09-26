"""Eksperimen probe: bisakah laya BELAJAR pola penalaran komparatif ("pedas" -> "cabai")?

Latar: test_pedas_repro menunjukkan semua varian laya (baseline/v1/v3/v4) gagal yakin
pada pertanyaan "yang paling X" — sementara Jev 7/7 benar. Pertanyaan pengguna: apakah
laya tidak bisa mendekati Jev sama sekali untuk kasus ini?

Desain eksperimen:
1. Bangun data sintetis comparative-choice: pertanyaan memuat sifat (properti),
   TEPAT SATU opsi berisi konsep jembatan di kriterianya (pedas -> cabai/lada/sambal).
   8 jembatan dilatih, 4 jembatan HOLD-OUT (tidak pernah muncul di training).
2. Fine-tune cepat dari runs/laya-idjvsuen-v4 (2 epoch, menit-an) TANPA replay ->
   model probe (bukan kandidat release; forgetting memang diharapkan).
3. Evaluasi:
   a. 7 varian test_pedas_repro asli (jembatan pedas DILATIH, item tak dilihat),
   b. varian jembatan hold-out (uji generalisasi vs hafalan),
   c. sanity sentimen/intent (mengukur seberapa parah forgetting).

Penggunaan:
  python scripts/16_comparative_probe.py            # data + train + evaluasi
  python scripts/16_comparative_probe.py --skip-train  # evaluasi ulang saja
"""
import argparse
import os
import random
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT_DIR = os.path.join(ROOT, "runs", "probe-comparative")
DATA_PATH = os.path.join(ROOT, "data", "processed", "train_items_probe.pt")

# jembatan properti -> konsep yang boleh muncul di kriteria opsi BENAR
BRIDGES_TRAINED = {
    "pedas": ["cabai", "lada", "sambal"],
    "manis": ["gula", "madu"],
    "asin": ["garam"],
    "dingin": ["es", "beku"],
    "mahal": ["harga selangit", "miliaran"],
    "cepat": ["sekilas", "kilat"],
    "kuat": ["besi", "baja"],
    "ramai": ["keramaian", "penuh penonton"],
}
BRIDGES_HELDOUT = {
    "pahit": ["kopi", "pare"],
    "asam": ["jeruk nipis", "cuka"],
    "lembut": ["sutra", "bulu"],
    "berat": ["timah", "batu"],
}
PROP_EN = {"pedas": "spicy", "manis": "sweet", "asin": "salty", "dingin": "cold",
           "mahal": "expensive", "cepat": "fast", "kuat": "strong", "ramai": "crowded",
           "pahit": "bitter", "asam": "sour", "lembut": "soft", "berat": "heavy"}

Q_TEMPLATES = [
    "Makanan apa yang paling {p}?",
    "Minuman apa yang paling {p}?",
    "Mana yang paling {p}?",
    "Pilih yang paling {p}.",
    "Yang paling {p} yang mana?",
    "Which one is the most {pe}?",
    "Which of these is the most {pe}?",
]
# deskripsi netral untuk distraktor (tidak memuat konsep jembatan pertanyaan aktif)
NEUTRAL_DESCS = [
    "ayam goreng krispi", "nasi putih hangat", "kerupuk udang renyah",
    "es teh segar", "roti bakar cokelat", "salad buah potong",
    "mie instan kuah", "pisang rebus", "baju katun biasa",
    "tas kain sekolah", "buku tulis bersampul karton", "gelas plastik bening",
    "kacamata hitam murah", "sepeda lipat bekas", "kursi kayu jati",
    "gelang benang", "topi jerami", "sandal karet",
]
OBJECTS = [
    "bakwan", "seblak", "pecel", "rujak", "sate", "gado-gado", "sayur asem",
    "es cincau", "wedang jahe", "sorbet", "sup", "martabak", "durian", "mangga",
    "kopi tubruk", "jus alpukat", "teh tarik", "cokelat panas",
    "meja", "tikar", "tenda", "kotak", "kabel", "mesin", "kompor",
]
# pengisi netral untuk distraktor ber-pola "dengan banyak ..." (mode balanced)
FILLERS = ["lalapan", "buah segar", "keju", "sayuran", "sirup merah", "kerupuk"]


def build_item(tok, seq_cfg, state, crit, gold_key, instructions):
    from laya.common import build_sequence, render_options, QTYPES

    keys = list(crit.keys())
    gold_idx = keys.index(gold_key)
    target = [0.0] * len(keys)
    target[gold_idx] = 1.0
    q = {"t": "choice", "ins": instructions, "crit": crit}
    k = len(render_options(q))
    seq, markers = build_sequence(tok, state, q, seq_cfg["max_len"], seq_cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES["choice"],
            "target": target, "label": gold_idx, "state": state,
            "meta": {"task": "comparative_probe", "gold": gold_key, "synthetic": True}}


def gen_data(tok, seq_cfg, rng, per_bridge=400, balanced=False):
    items = []
    bridges = list(BRIDGES_TRAINED.items())
    correct_tmpl = ["{obj} dengan banyak {c}", "{obj} balutan {c}", "{obj} penuh {c}"]
    for prop, concepts in bridges:
        forbidden = set(concepts)
        n = 0
        while n < per_bridge:
            tmpl = rng.choice(Q_TEMPLATES)
            state = tmpl.format(p=prop, pe=PROP_EN[prop])
            concept = rng.choice(concepts)
            obj = rng.choice(OBJECTS)
            # balanced: opsi benar hanya separuh waktu memakai pola "dengan banyak",
            # sehingga frasa itu tidak bisa jadi shortcut yang menentukan emas.
            correct = (rng.choice(correct_tmpl) if balanced
                       else "{obj} dengan banyak {c}").format(obj=obj, c=concept)
            if balanced and rng.random() < 0.5:
                descs = [f"{rng.choice(OBJECTS)} dengan banyak {rng.choice(FILLERS)}"
                         for _ in range(2)]
            else:
                descs = rng.sample(NEUTRAL_DESCS, 2)
            if any(f in d for d in descs for f in forbidden):
                continue  # distraktor tak boleh memuat konsep jembatan aktif
            keys = [f"pilihan {chr(97 + i)}" for i in range(3)]
            gold_key = rng.choice(keys)
            others = list(descs)
            crit = {k: (correct if k == gold_key else others.pop(0)) for k in keys}
            it = build_item(tok, seq_cfg, state, crit, gold_key,
                            "Choose the option that best answers the question.")
            if it:
                items.append(it)
                n += 1
    rng.shuffle(items)
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-bridge", type=int, default=400)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--balanced", action="store_true",
                   help="mode v2: distraktor ikut memuat pola 'dengan banyak {filler}', "
                        "opsi benar hanya separuh waktu memakai pola itu "
                        "(menetralkan shortcut frasa)")
    p.add_argument("--skip-train", action="store_true",
                   help="lewati pembangunan data + training (eval ulang saja)")
    args = p.parse_args()

    out_dir = OUT_DIR + "-v2" if args.balanced else OUT_DIR

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))

    if not args.skip_train:
        from transformers import AutoTokenizer
        from laya.agent import _fix_tokenizer_config

        v4 = os.path.join(ROOT, "runs", "laya-idjvsuen-v4")
        _fix_tokenizer_config(v4)
        tok = AutoTokenizer.from_pretrained(os.path.join(v4, "tokenizer"))
        rng = random.Random(20260926)
        items = gen_data(tok, {"max_len": 768, "head_max_len": 512}, rng,
                         args.per_bridge, balanced=args.balanced)
        torch.save(items, DATA_PATH)
        print(f"[data] {len(items)} item comparative probe -> {DATA_PATH}")

        cmd = [sys.executable, os.path.join(ROOT, "scripts", "04_train.py"),
               "--model", v4, "--data", DATA_PATH, "--out", out_dir,
               "--model-name", "probe-comparative",
               "--epochs", str(args.epochs), "--micro-batch", "8", "--grad-accum", "8",
               "--optim", "adamw8bit"]
        print("[train]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    # ---- evaluasi: varian asli + jembatan hold-out + sanity ----
    import test_pedas_repro as tpr

    # KONTROL penentu: distraktor sengaja memuat pola "dengan banyak" TANPA konsep
    # jembatan, sedangkan opsi benar memuat konsep TANPA pola "banyak".
    # Kalau model tetap memilih seblak -> hubungan properti-konsep sungguhan;
    # kalau memilih distraktor -> yang dipelajari cuma shortcut frasa "dengan banyak".
    controls = [
        ("CTRL-A banyak-shortcut, jembatan dilatih (pedas)",
         "Makanan apa yang paling pedas?", tpr.INS_GENERIC,
         {"pecel kacang": "sayur rebus bumbu kacang",
          "ramesan": "gado-gado dengan banyak lalapan",
          "seblak": "kerupuk rebus balutan sambal"}),
        ("CTRL-B banyak-shortcut, jembatan HOLD-OUT (pahit)",
         "Minuman apa yang paling pahit?", tpr.INS_GENERIC,
         {"pecel kacang": "es teh manis susu",
          "ramesan": "rujakan dengan banyak buah segar",
          "seblak": "seduhan kopi murni tanpa gula"}),
        ("CTRL-C konsep dilatih lain (manis->madu), tanpa 'banyak'",
         "Mana yang paling manis?", tpr.INS_GENERIC,
         {"pecel kacang": "kerupuk asin gurih",
          "ramesan": "es campur dengan banyak sirup merah",
          "seblak": "sarangan madu hutan asli"}),
    ]

    heldout_variants = []
    rng = random.Random(7)
    for prop, concepts in BRIDGES_HELDOUT.items():
        concept = rng.choice(concepts)
        d1, d2 = rng.sample(NEUTRAL_DESCS, 2)
        crit = {"pecel kacang": d1, "ramesan": d2,
                "seblak": f"kerupuk rebus dengan banyak {concept}"}
        heldout_variants.append(
            (f"HOLD-OUT: {prop}->{concept}", f"Makanan apa yang paling {prop}?",
             tpr.INS_GENERIC, crit))
    sanity = [
        ("SANITY sentimen (harus: negative)", "Pelayanane elek tenan, aku ora arep balik maneh",
         tpr.INS_GENERIC,
         {"negative": "the text expresses a negative opinion, complaint, or dissatisfaction",
          "neutral": "the text is neutral, factual, or mixed",
          "positive": "the text expresses a positive opinion, praise, or satisfaction"}),
        ("SANITY topik (harus: sports)", "Timnas berjuang di laga pamungkas kualifikasi",
         "Classify the main topic of the text into the most likely category.",
         {"sports": "about sports, athletes, games, or competitions",
          "health": "about health, medicine, disease, or the human body",
          "travel": "about travel, tourism, destinations, or trips"}),
    ]
    tpr.VARIANTS = list(tpr.VARIANTS) + heldout_variants + controls + sanity
    tpr.run_model("PROBE comparative (fine-tune dari v4)", out_dir)


if __name__ == "__main__":
    main()
