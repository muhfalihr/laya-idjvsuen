"""Repro input dari screenshot user (POST /predict, Debian) pada baseline / v1 / v3 / v4
+ perbandingan Jev (OpenRouter /api/alpha/decisions, ~typesafe/jev-latest).

Payload persis:
  text         : "Makanan apa yang paling pedas?"
  instructions : "Classify the user utterance into the most appropriate category."
  criteria     : pecel kacang / ramesan / seblak (dengan deskripsi)

Menjalankan jalur inference identik 05_eval.py (forward + temperatur config checkpoint)
dan mencetak probabilitas per opsi. Varian tambahan untuk diagnosis.

Jev dipanggil dengan skema identik (state + questions.choice + criteria); kunci API
dibaca dari .env (OPENROUTER_API_KEY). Bagian Jev dilewati lembut bila kunci/requests
tidak tersedia.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from hflocal import local_snapshot  # noqa: E402

JEV_API_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL_ID = "~typesafe/jev-latest"

TEXT = "Makanan apa yang paling pedas?"
INS_GENERIC = "Classify the user utterance into the most appropriate category."
CRIT = {
    "pecel kacang": "pecel hanya bumbu kacang tanpa cabai",
    "ramesan": "ayam goreng krispi",
    "seblak": "kerupuk rebus dengan banyak cabai",
}

# varian diagnosis: instruksi selaras pertanyaan, tanpa kriteria, bahasa Inggris
VARIANTS = [
    ("screenshot (persis)", TEXT, INS_GENERIC, CRIT),
    ("instruksi selaras pertanyaan", TEXT, "Answer the question by choosing the option that best matches what the question asks (the spiciest food).", CRIT),
    ("instruksi tanya-jawab id", TEXT, "Pilih jawaban yang paling sesuai dengan pertanyaan.", CRIT),
    ("tanpa kriteria (label saja)", TEXT, INS_GENERIC, {"pecel kacang": None, "ramesan": None, "seblak": None}),
    ("pertanyaan versi en", "Which food is the spiciest?", INS_GENERIC, CRIT),
    # uji hipotesis pencocokan leksikal: kata kunci "pedas" dimasukkan ke kriteria seblak
    ("kriteria seblak dgn kata 'pedas'", TEXT, INS_GENERIC, {
        "pecel kacang": "pecel hanya bumbu kacang tanpa cabai",
        "ramesan": "ayam goreng krispi",
        "seblak": "makanan paling pedas, kerupuk rebus dengan banyak cabai"}),
    # uji hipotesis overlap kata: pertanyaan menyebut "cabai" yang ada di kriteria seblak
    ("pertanyaan menyebut 'cabai'", "Makanan apa yang paling banyak cabainya?", INS_GENERIC, CRIT),
]


def run_model(tag, model_dir):
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, clamp_temperature, build_sequence, QTYPES

    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    temps = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    seq_cfg = cfg.get("seq_cfg") or json.load(open(os.path.join(model_dir, "question_defs.json")))["seq_cfg"] \
        if os.path.exists(os.path.join(model_dir, "question_defs.json")) else {"max_len": 768, "head_max_len": 512}
    seq_cfg = seq_cfg if isinstance(seq_cfg, dict) and "max_len" in seq_cfg else {"max_len": 768, "head_max_len": 512}

    print(f"\n=== {tag} ({model_dir}) temps={temps} device={device.type} ===")
    pad_id = tok.pad_token_id
    with torch.no_grad():
        for name, state, ins, crit in VARIANTS:
            q = {"t": "choice", "ins": ins, "crit": crit}
            seq, markers = build_sequence(tok, state, q, seq_cfg["max_len"], seq_cfg["head_max_len"])
            ids = torch.tensor([seq], device=device)
            att = torch.ones_like(ids)
            mpos = torch.tensor([markers + [0] * (64 - len(markers))], device=device)
            mmask = torch.zeros((1, 64), dtype=torch.bool, device=device)
            mmask[0, : len(markers)] = True
            qt = torch.tensor([QTYPES["choice"]], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _ = model(ids, att, mpos, mmask, qt)
            z = logits[0, : len(markers)].float().cpu().numpy() / temps[QTYPES["choice"]]
            p = np.exp(z - z.max())
            p = p / p.sum()
            keys = list(crit.keys())
            probs = "  ".join(f"{k}={v:.4f}" for k, v in zip(keys, p))
            pick = keys[int(p.argmax())]
            ok = "BENAR (seblak)" if pick == "seblak" else f"SALAH (pilih {pick})"
            print(f"  [{name}] {probs}  -> {ok}")


def load_env_file():
    """Baca ROOT/.env sederhana (KEY=VALUE) tanpa menimpa env yang sudah ada."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def run_jev(model_id=JEV_MODEL_ID):
    """Varian yang sama dikirim ke Jev (skema kompatibel laya), hasil dicetak berdampingan."""
    load_env_file()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n=== Jev: DILEWATI (OPENROUTER_API_KEY tidak ada di env/.env) ===")
        return
    try:
        import requests
    except ImportError:
        print("\n=== Jev: DILEWATI (requests tidak terpasang) ===")
        return

    print(f"\n=== Jev ({model_id}) via {JEV_API_URL} ===")
    session = requests.Session()
    headers = {
        "Authorization": "Bearer %s" % os.environ["OPENROUTER_API_KEY"],
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/muhfalihr/laya-idjvsuen",
        "X-Title": "laya-idjvsuen pedas repro",
    }
    total_cost = 0.0
    for name, state, ins, crit in VARIANTS:
        # kriteria None (label saja) -> pakai nama opsi sebagai deskripsi untuk API
        crit_api = {k: (v if isinstance(v, str) and v else k) for k, v in crit.items()}
        body = {"model": model_id, "state": state,
                "questions": {"q": {"type": "choice", "instructions": ins,
                                    "criteria": crit_api}}}
        try:
            t0 = time.time()
            r = session.post(JEV_API_URL, headers=headers, timeout=120, json=body)
            dt = time.time() - t0
            r.raise_for_status()
            d = r.json()
            a = d["answers"]["q"]
            probs_d = a.get("probabilities", {}) or {}
            keys = list(crit.keys())
            p = np.array([float(probs_d.get(k, 0.0)) for k in keys])
            if p.sum() > 0:
                p = p / p.sum()
            pick = a.get("choice") or keys[int(p.argmax())]
            cost = d.get("usage", {}).get("cost")
            if cost is not None:
                total_cost += float(cost)
            probs = "  ".join(f"{k}={v:.4f}" for k, v in zip(keys, p))
            ok = "BENAR (seblak)" if pick == "seblak" else f"SALAH (pilih {pick})"
            print(f"  [{name}] {probs}  -> {ok}  ({dt:.2f}s, conf={a.get('confidence')})")
        except Exception as e:
            print(f"  [{name}] GAGAL: {type(e).__name__}: {e}")
    print(f"  total biaya Jev: ${total_cost:.6f}")


if __name__ == "__main__":
    run_model("baseline laya-multilingual", local_snapshot("convaiinnovations/laya-multilingual", ROOT))
    run_model("laya-idjvsuen-v1", os.path.join(ROOT, "runs", "laya-idjvsuen-v1"))
    run_model("laya-idjvsuen-v3", os.path.join(ROOT, "runs", "laya-idjvsuen-v3"))
    run_model("laya-idjvsuen-v4", os.path.join(ROOT, "runs", "laya-idjvsuen-v4"))
    run_jev()
