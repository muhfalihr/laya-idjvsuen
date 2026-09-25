"""Jalankan laya-idjvsuen-v1 pada pesan ticket produksi (hasil 09_preprocess).

Dua tugas:
1. sentimen (3 kelas) - in-distribution, format pertanyaan persis training (question_defs.json)
   pada sampel besar, stratified per lang_guess.
2. intent (60 kelas MASSIVE) - sampel kecil, untuk melihat pemetaan ke permintaan nyata.

Output: data/processed/tickets_eval.json + ringkasan + contoh per kelas.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
MODEL = os.path.join(ROOT, "runs", "laya-idjvsuen-v1")


def stratified_sample(rows, n, seed=42):
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang_guess"]].append(r)
    for v in by_lang.values():
        rng.shuffle(v)
    quota = {k: round(n * len(v) / len(rows)) for k, v in by_lang.items()}
    out = []
    for k, v in by_lang.items():
        out.extend(v[: quota[k]])
    rng.shuffle(out)
    return out[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-sentiment", type=int, default=3000)
    p.add_argument("--n-intent", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=os.path.join(PROC, "tickets_eval.json"))
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    import laya

    qdefs = json.load(open(os.path.join(args.model, "question_defs.json"), encoding="utf-8"))
    agent = laya.load(args.model)
    rows = [json.loads(l) for l in open(os.path.join(PROC, "tickets_user.jsonl"), encoding="utf-8")]
    print(f"{len(rows)} pesan user dimuat")

    results = {}

    # ---------- 1. sentimen ----------
    sample = stratified_sample(rows, args.n_sentiment)
    q_senti = {"s": {"type": "choice", "instructions": qdefs["sentiment"]["instructions"],
                     "criteria": qdefs["sentiment"]["criteria"]}}
    preds = agent.predict_batch([r["text"] for r in sample], q_senti, batch_size=args.batch_size)
    agg = {"per_label": Counter(), "per_lang": defaultdict(Counter), "conf": []}
    examples = defaultdict(list)
    for r, res in zip(sample, preds):
        a = res["answers"]["s"]
        label, conf = a["choice"], a["answer_confidence"]
        agg["per_label"][label] += 1
        agg["per_lang"][r["lang_guess"]][label] += 1
        agg["conf"].append(conf)
        if len(examples[label]) < 8 and len(r["text"]) > 25:
            examples[label].append({"text": r["text"][:120], "conf": conf,
                                    "lang": r["lang_guess"]})
    n = len(sample)
    conf_sorted = sorted(agg["conf"])
    results["sentiment"] = {
        "n": n,
        "per_label": dict(agg["per_label"]),
        "per_label_pct": {k: round(v / n * 100, 1) for k, v in agg["per_label"].items()},
        "per_lang": {k: dict(v) for k, v in agg["per_lang"].items()},
        "mean_conf": round(sum(agg["conf"]) / n, 3),
        "conf_p50": conf_sorted[n // 2],
        "conf_p10": conf_sorted[n // 10],
        "low_conf_share": round(sum(1 for c in agg["conf"] if c < 0.6) / n * 100, 1),
        "examples": {k: v for k, v in examples.items()},
    }

    # ---------- 2. intent ----------
    sample_i = stratified_sample(rows, args.n_intent, seed=7)
    q_int = {"i": {"type": "choice", "instructions": qdefs["intent"]["instructions"],
                   "criteria": qdefs["intent"]["criteria"]}}
    preds_i = agent.predict_batch([r["text"] for r in sample_i], q_int, batch_size=args.batch_size)
    intents = Counter()
    conf_i = []
    ex_i = defaultdict(list)
    for r, res in zip(sample_i, preds_i):
        a = res["answers"]["i"]
        intents[a["choice"]] += 1
        conf_i.append(a["answer_confidence"])
        if len(ex_i[a["choice"]]) < 3 and len(r["text"]) > 25:
            ex_i[a["choice"]].append({"text": r["text"][:120], "conf": a["answer_confidence"]})
    ni = len(sample_i)
    results["intent"] = {
        "n": ni,
        "top15": intents.most_common(15),
        "mean_conf": round(sum(conf_i) / ni, 3),
        "share_top15": round(sum(c for _, c in intents.most_common(15)) / ni * 100, 1),
        "examples": {k: v for k, v in ex_i.items()},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    s = results["sentiment"]
    print(f"\n=== SENTIMEN ({s['n']} pesan) ===")
    for k, v in s["per_label_pct"].items():
        print(f"  {k:<9} {v:>5}%  (contoh: {s['per_label'][k]})")
    print(f"  mean confidence: {s['mean_conf']} | p10: {s['conf_p10']} | "
          f"share conf<0.6: {s['low_conf_share']}%")
    print("  per bahasa:", {k: dict(v) for k, v in s["per_lang"].items()})
    print("\n  contoh negative paling percaya:")
    for e in sorted(examples.get("negative", []), key=lambda x: -x["conf"])[:4]:
        print(f"    [{e['conf']:.2f}|{e['lang']}] {e['text']}")

    i = results["intent"]
    print(f"\n=== INTENT MASSIVE ({i['n']} pesan, probe) ===")
    for name, c in i["top15"]:
        print(f"  {name:<28} {c:>4}")
    print(f"  mean confidence: {i['mean_conf']} | top-15 mencakup {i['share_top15']}% prediksi")
    print(f"\nhasil tersimpan: {args.out}")


if __name__ == "__main__":
    main()
