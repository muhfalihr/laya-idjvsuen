"""Bandingkan hasil evaluasi baseline vs fine-tuned menjadi tabel markdown + JSON.
Menghasilkan dua bahasa sekaligus: docs/HASIL-RETRAINING.md (id) dan .en.md (en).

Penggunaan:
  python scripts/06_compare.py --baseline data/processed/eval_baseline.json \
      --finetuned data/processed/eval_finetuned.json
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SET_ORDER = ["intent_id", "intent_en", "intent_jv", "intent_sun",
             "sentiment_id", "sentiment_en", "sentiment_jv", "sentiment_sun",
             "cs_id_en", "cs_id_jv", "cs_id_sun"]
NICE = {
    "intent_id": ("Intent Indonesia (MASSIVE 60-kelas)", "Indonesian intent (MASSIVE 60-class)"),
    "intent_en": ("Intent Inggris (MASSIVE 60-kelas)", "English intent (MASSIVE 60-class)"),
    "intent_jv": ("Intent Jawa (terj. NLLB)", "Javanese intent (NLLB-translated)"),
    "intent_sun": ("Intent Sunda (terj. NLLB)", "Sundanese intent (NLLB-translated)"),
    "sentiment_id": ("Sentimen Indonesia (NusaX)", "Indonesian sentiment (NusaX)"),
    "sentiment_en": ("Sentimen Inggris (NusaX)", "English sentiment (NusaX)"),
    "sentiment_jv": ("Sentimen Jawa (NusaX)", "Javanese sentiment (NusaX)"),
    "sentiment_sun": ("Sentimen Sunda (NusaX)", "Sundanese sentiment (NusaX)"),
    "cs_id_en": ("Code-switch id+en (sintetis)", "Code-switch id+en (synthetic)"),
    "cs_id_jv": ("Code-switch id+jv (sintetis)", "Code-switch id+jv (synthetic)"),
    "cs_id_sun": ("Code-switch id+sun (sintetis)", "Code-switch id+su (synthetic)"),
}
T = {
    "id": {"title": "# Hasil Retraining Laya — Baseline vs Fine-tuned",
           "lang_switch": "**Bahasa**: **Indonesia** | [English](HASIL-RETRAINING.en.md)",
           "base": "Baseline", "ft": "Fine-tuned",
           "cols": ("| Test set | n | Akurasi base | Akurasi FT | Δ | ECE base | ECE FT |",
                    "|---|---|---|---|---|---|---|"),
           "agg_title": "## Rata-rata per tugas",
           "agg_cols": ("| Tugas | Akurasi base | Akurasi FT | Δ |", "|---|---|---|---|"),
           "tasks": (("Intent (4 bahasa)", "intent"), ("Sentimen (4 bahasa)", "sentiment"),
                     ("Code-switch", "cs")),
           "out": os.path.join(ROOT, "docs", "HASIL-RETRAINING.md")},
    "en": {"title": "# Laya Retraining Results — Baseline vs Fine-tuned",
           "lang_switch": "**Bahasa / Language**: [Indonesia](HASIL-RETRAINING.md) | **English**",
           "base": "Baseline", "ft": "Fine-tuned",
           "cols": ("| Test set | n | Baseline acc | FT acc | Δ | Baseline ECE | FT ECE |",
                    "|---|---|---|---|---|---|---|"),
           "agg_title": "## Per-task averages",
           "agg_cols": ("| Task | Baseline acc | FT acc | Δ |", "|---|---|---|---|"),
           "tasks": (("Intent (4 languages)", "intent"), ("Sentiment (4 languages)", "sentiment"),
                     ("Code-switching", "cs")),
           "out": os.path.join(ROOT, "docs", "HASIL-RETRAINING.en.md")},
}


def build(base, ft, lang, idx):
    t = T[lang]
    # map path lokal run -> repo HuggingFace untuk label dokumen
    hf_map = {"runs/laya-idjvsuen-v1": "faall7479/laya-idjvsuen-v1",
              "runs\\laya-idjvsuen-v1": "faall7479/laya-idjvsuen-v1",
              "runs/laya-idjvsuen-v3": "faall7479/laya-idjvsuen-v3",
              "runs\\laya-idjvsuen-v3": "faall7479/laya-idjvsuen-v3",
              "runs/laya-idjvsuen-v4": "faall7479/laya-idjvsuen-v4",
              "runs\\laya-idjvsuen-v4": "faall7479/laya-idjvsuen-v4"}
    ft_model = ft.get("_summary", {}).get("model", "faall7479/laya-idjvsuen-v1")
    lines = [t["title"], "", t["lang_switch"], "",
             f"- {t['base']}: `{base.get('_summary', {}).get('model', 'laya-multilingual')}`",
             f"- {t['ft']}: `{hf_map.get(ft_model, ft_model)}`",
             "", t["cols"][0], t["cols"][1]]
    for name in SET_ORDER:
        if name not in base or name not in ft:
            continue
        b, f = base[name], ft[name]
        delta = f["accuracy"] - b["accuracy"]
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {NICE[name][idx]} | {b['n']} | {b['accuracy']:.3f} | "
                     f"{f['accuracy']:.3f} | {sign}{delta:.3f} | {b['ece']:.3f} | {f['ece']:.3f} |")

    def agg(prefix):
        sel_b = [base[k]["accuracy"] for k in SET_ORDER if k.startswith(prefix) and k in base]
        sel_f = [ft[k]["accuracy"] for k in SET_ORDER if k.startswith(prefix) and k in ft]
        return (sum(sel_b) / len(sel_b) if sel_b else 0,
                sum(sel_f) / len(sel_f) if sel_f else 0)

    lines += ["", t["agg_title"], "", t["agg_cols"][0], t["agg_cols"][1]]
    for label, prefix in t["tasks"]:
        b, f = agg(prefix)
        lines.append(f"| {label} | {b:.3f} | {f:.3f} | {'+' if f - b >= 0 else ''}{f - b:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=os.path.join(ROOT, "data", "processed", "eval_baseline.json"))
    p.add_argument("--finetuned", default=os.path.join(ROOT, "data", "processed", "eval_finetuned.json"))
    args = p.parse_args()

    base = json.load(open(args.baseline, encoding="utf-8"))
    ft = json.load(open(args.finetuned, encoding="utf-8"))

    for lang, idx in (("id", 0), ("en", 1)):
        text = build(base, ft, lang, idx)
        os.makedirs(os.path.dirname(T[lang]["out"]), exist_ok=True)
        with open(T[lang]["out"], "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        print(f"tersimpan / saved: {T[lang]['out']}\n")


if __name__ == "__main__":
    main()
