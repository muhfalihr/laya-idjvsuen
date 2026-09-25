"""Bandingkan hasil evaluasi baseline vs fine-tuned jadi satu tabel markdown + JSON.

Penggunaan:
  python scripts/06_compare.py --baseline data/processed/eval_baseline.json \
      --finetuned data/processed/eval_finetuned.json --out HASIL-RETRAINING.md
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SET_ORDER = ["intent_id", "intent_en", "intent_jv", "intent_sun",
             "sentiment_id", "sentiment_en", "sentiment_jv", "sentiment_sun",
             "cs_id_en", "cs_id_jv", "cs_id_sun"]
NICE = {
    "intent_id": "Intent Indonesia (MASSIVE 60-kelas)",
    "intent_en": "Intent Inggris (MASSIVE 60-kelas)",
    "intent_jv": "Intent Jawa (terj. NLLB)",
    "intent_sun": "Intent Sunda (terj. NLLB)",
    "sentiment_id": "Sentimen Indonesia (NusaX)",
    "sentiment_en": "Sentimen Inggris (NusaX)",
    "sentiment_jv": "Sentimen Jawa (NusaX)",
    "sentiment_sun": "Sentimen Sunda (NusaX)",
    "cs_id_en": "Code-switch id+en (sintetis)",
    "cs_id_jv": "Code-switch id+jv (sintetis)",
    "cs_id_sun": "Code-switch id+sun (sintetis)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=os.path.join(ROOT, "data", "processed", "eval_baseline.json"))
    p.add_argument("--finetuned", default=os.path.join(ROOT, "data", "processed", "eval_finetuned.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "HASIL-RETRAINING.md"))
    args = p.parse_args()

    base = json.load(open(args.baseline, encoding="utf-8"))
    ft = json.load(open(args.finetuned, encoding="utf-8"))

    lines = [
        "# Hasil Retraining Laya — Baseline vs Fine-tuned",
        "",
        f"- Baseline: `{base.get('_summary', {}).get('model', 'laya-multilingual')}`",
        f"- Fine-tuned: `{ft.get('_summary', {}).get('model', 'runs/laya-idjvsuen-v1')}`",
        "",
        "| Test set | n | Akurasi base | Akurasi FT | Δ | ECE base | ECE FT |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for name in SET_ORDER:
        if name not in base or name not in ft:
            continue
        b, f = base[name], ft[name]
        delta = f["accuracy"] - b["accuracy"]
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {NICE.get(name, name)} | {b['n']} | {b['accuracy']:.3f} | "
                     f"{f['accuracy']:.3f} | {sign}{delta:.3f} | {b['ece']:.3f} | {f['ece']:.3f} |")
        rows.append({"set": name, "base_acc": b["accuracy"], "ft_acc": f["accuracy"],
                     "delta": round(delta, 4), "base_ece": b["ece"], "ft_ece": f["ece"]})

    def agg(prefix):
        sel_b = [base[k]["accuracy"] for k in SET_ORDER if k.startswith(prefix) and k in base]
        sel_f = [ft[k]["accuracy"] for k in SET_ORDER if k.startswith(prefix) and k in ft]
        return (sum(sel_b) / len(sel_b) if sel_b else 0, sum(sel_f) / len(sel_f) if sel_f else 0)

    lines += ["", "## Rata-rata per tugas", "",
              "| Tugas | Akurasi base | Akurasi FT | Δ |", "|---|---|---|---|"]
    for label, prefix in (("Intent (4 bahasa)", "intent"), ("Sentimen (4 bahasa)", "sentiment"),
                          ("Code-switch", "cs")):
        b, f = agg(prefix)
        lines.append(f"| {label} | {b:.3f} | {f:.3f} | {'+' if f-b>=0 else ''}{f-b:.3f} |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\ntersimpan: {args.out}")


if __name__ == "__main__":
    main()
