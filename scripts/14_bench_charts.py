"""Buat chart benchmark bergaya model card resmi Laya (untuk README GitHub & HF).

1. bench_overall.png   - bar 4 model pada 300 sampel identik: accuracy, macro-F1, ECE
2. bench_perclass.png  - heatmap matriks per-kelas: model x 12 kategori
3. bench_languages.png - sebelum/sesudah fine-tune pada 15 test set bahasa/tugas

Data: data/processed/eval_*.json + prediksi per-item (laya via forward, Jev dari dump).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
ASSETS = os.path.join(ROOT, "docs", "assets")

MODELS = [
    ("laya-multilingual", "laya-multilingual\n(no fine-tune)", "#9e9e9e"),
    ("v1", "laya-idjvsuen-v1\n(id/jv/su/en)", "#8ecae6"),
    ("jev", "Jev 1.13\n(OpenRouter)", "#ffb703"),
    ("v3", "laya-idjvsuen-v3\n(domain fine-tune)", "#219ebc"),
]


def per_item_preds(model_dir, items):
    """Prediksi indeks kelas per item (forward path identik 05_eval)."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, clamp_temperature

    _fix_tokenizer_config(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.to("cuda").eval()
    temps = [clamp_temperature(t) for t in cfg.get("temperature", [1, 1, 1])]
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            L = max(len(it["ids"]) for it in chunk)
            kmax = max(len(it["markers"]) for it in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            mpos = torch.zeros((len(chunk), kmax), dtype=torch.long)
            mmask = torch.zeros((len(chunk), kmax), dtype=torch.bool)
            for j, it in enumerate(chunk):
                ids[j, :len(it["ids"])] = torch.tensor(it["ids"])
                att[j, :len(it["ids"])] = 1
                mpos[j, :len(it["markers"])] = torch.tensor(it["markers"])
                mmask[j, :len(it["markers"])] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids.to("cuda"), att.to("cuda"), mpos.to("cuda"),
                                  mmask.to("cuda"),
                                  torch.tensor([it["qtype"] for it in chunk]).to("cuda"))
            lg = logits.float().cpu().numpy()
            for j, it in enumerate(chunk):
                z = lg[j, :len(it["markers"])] / temps[it["qtype"]]
                preds.append(int(z.argmax()))
    del model
    torch.cuda.empty_cache()
    return preds


def chart_overall():
    from matplotlib.patches import Patch

    acc, f1, ece, labels, colors = [], [], [], [], []
    for key, label, color in MODELS:
        fn = "eval_jev.json" if key == "jev" else f"eval_{key}_ticket300.json"
        if key == "laya-multilingual":
            fn = "eval_base_ticket300.json"
        d = json.load(open(os.path.join(PROC, fn)))["ticket_intent"]
        acc.append(d["accuracy"] * 100)
        f1.append(d["macro_f1"] * 100)
        ece.append(d["ece"] * 100)
        labels.append(label.replace("\n", " "))
        colors.append(color)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, vals, title in zip(axes, [acc, f1, ece],
                               ["Accuracy (%)", "Macro-F1 (%)", "ECE — calibration error (%)"]):
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(["1", "2", "3", "4"], fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.18 + 2
        ax.set_ylim(max(0, lo - pad), hi + pad)
        for xi, v in enumerate(vals):
            ax.text(xi, v + pad * 0.06, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Ticket-domain 12-category classification — identical 300 real samples",
                 fontsize=11, fontweight="bold")
    handles = [Patch(facecolor=c, label=f"{i + 1} — {l}") for i, (l, c) in enumerate(zip(labels, colors))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(ASSETS, "bench_overall.png"), dpi=160)
    plt.close(fig)


def chart_perclass():
    ts = torch.load(os.path.join(PROC, "test_sets_tickets300.pt"), weights_only=False)["ticket_intent"]
    items, names = ts["items"], ts["label_names"]
    y = [it["label"] for it in items]

    preds = {}
    preds["v3"] = per_item_preds(os.path.join(ROOT, "runs", "laya-idjvsuen-v3"), items)
    preds["v1"] = per_item_preds(os.path.join(ROOT, "runs", "laya-idjvsuen-v1"), items)
    preds["laya-multilingual"] = per_item_preds(
        os.path.join(ROOT, "models", "convaiinnovations__laya-multilingual"), items)
    dump = [json.loads(l) for l in open(os.path.join(PROC, "bench_jev_items.jsonl"), encoding="utf-8")]
    preds["jev"] = [names.index(r["pred"]) if r["pred"] in names else -1 for r in dump]

    rows = ["laya-multilingual", "v1", "jev", "v3"]
    row_labels = ["laya-multilingual", "laya-idjvsuen-v1", "Jev 1.13", "laya-idjvsuen-v3"]
    mat = np.zeros((len(rows), len(names)))
    for ri, r in enumerate(rows):
        for ci in range(len(names)):
            sel = [i for i, t in enumerate(y) if t == ci]
            mat[ri, ci] = (sum(1 for i in sel if preds[r][i] == y[i]) / len(sel) * 100) if sel else np.nan

    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for ri in range(len(rows)):
        for ci in range(len(names)):
            v = mat[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "n/a", ha="center", va="center", fontsize=8, color="#777")
            else:
                ax.text(ci, ri, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if rows[ri] == "v3" else "normal")
    ax.set_title("Per-category accuracy (%) — identical 300 samples", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="%")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "bench_perclass.png"), dpi=160)
    plt.close(fig)


def chart_languages():
    base = json.load(open(os.path.join(PROC, "eval_baseline.json"), encoding="utf-8"))
    ft = json.load(open(os.path.join(PROC, "eval_finetuned.json"), encoding="utf-8"))
    sets = ["intent_id", "intent_en", "intent_jv", "intent_sun",
            "sentiment_id", "sentiment_en", "sentiment_jv", "sentiment_sun",
            "cs_id_en", "cs_id_jv", "cs_id_sun"]
    nice = ["intent\nid", "intent\nen", "intent\njv", "intent\nsu",
            "senti\nid", "senti\nen", "senti\njv", "senti\nsu",
            "CS\nid+en", "CS\nid+jv", "CS\nid+su"]
    b = [base[k]["accuracy"] * 100 for k in sets]
    f = [ft[k]["accuracy"] * 100 for k in sets]
    x = np.arange(len(sets))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, b, 0.38, label="laya-multilingual (baseline)", color="#9e9e9e")
    ax.bar(x + 0.2, f, 0.38, label="laya-idjvsuen-v1 (fine-tuned)", color="#219ebc")
    for xi, (bv, fv) in zip(x, zip(b, f)):
        ax.text(xi - 0.2, bv + 1, f"{bv:.0f}", ha="center", fontsize=7.5)
        ax.text(xi + 0.2, fv + 1, f"{fv:.0f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Multilingual capability before vs after fine-tuning (id/jv/su/en + code-switching)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "bench_languages.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    chart_overall()
    chart_perclass()
    chart_languages()
    for f_ in ("bench_overall.png", "bench_perclass.png", "bench_languages.png"):
        p = os.path.join(ASSETS, f_)
        print(f_, os.path.getsize(p) // 1024, "KB")
