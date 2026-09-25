"""Benchmark model keputusan OpenRouter (mis. ~typesafe/jev-latest) pada test set laya.

Endpoint /api/alpha/decisions — skema permintaan kompatibel laya:
  {model, state, questions: {qid: {type: "choice", instructions, criteria}}}
Jawaban: answers.qid.{choice, probabilities, confidence}.

Metrik: accuracy, macro-F1, ECE (dari probabilities), parse/API failure, latensi,
total biaya (usage.cost). Output format eval_* agar bisa dibandingkan langsung.

Prasyarat: env OPENROUTER_API_KEY.

Penggunaan:
  python scripts/13_bench_openrouter.py --model-id "~typesafe/jev-latest" \
      --test-sets data/processed/test_sets_tickets.pt --set ticket_intent --n 300 --tag jev
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://openrouter.ai/api/alpha/decisions"


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def ece_score(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def call_one(session, model_id, state, instructions, criteria, max_retries=3):
    import requests

    headers = {
        "Authorization": "Bearer %s" % os.environ["OPENROUTER_API_KEY"],
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/muhfalihr/laya-idjvsuen",
        "X-Title": "laya-idjvsuen benchmark",
    }
    body = {
        "model": model_id,
        "state": state,
        "questions": {"q": {"type": "choice", "instructions": instructions,
                            "criteria": criteria}},
    }
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = session.post(API_URL, headers=headers, timeout=120, json=body)
            dt = time.time() - t0
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            a = d["answers"]["q"]
            usage = d.get("usage", {})
            return {"pred": a.get("choice"),
                    "conf": a.get("confidence"),
                    "resolved_model": d.get("model"),
                    "latency": round(dt, 2),
                    "cost": usage.get("cost", 0),
                    "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"pred": None, "conf": None, "resolved_model": None,
                        "latency": 0, "cost": 0, "tokens": 0,
                        "raw": str(e)[:100]}
            time.sleep(3 * (attempt + 1))
    return {"pred": None, "conf": None, "resolved_model": None, "latency": 0,
            "cost": 0, "tokens": 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="~typesafe/jev-latest")
    p.add_argument("--test-sets", default=os.path.join(ROOT, "data", "processed", "test_sets_tickets.pt"))
    p.add_argument("--set", default="ticket_intent")
    p.add_argument("--n", type=int, default=300, help="jumlah sampel (biaya API)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--tag", default="jev")
    args = p.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set env OPENROUTER_API_KEY dulu.")

    import requests

    ts = torch.load(args.test_sets, weights_only=False)[args.set]
    items = ts["items"][: args.n]
    keys = ts["label_names"]
    print(f"{len(items)} sampel | {len(keys)} opsi | model {args.model_id}")

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(
            lambda it: call_one(session, args.model_id, it["state"],
                                ts["instructions"], ts["criteria"]),
            items))

    y_true = [it["label"] for it in items]
    y_pred = [keys.index(r["pred"]) if r["pred"] in keys else -1 for r in results]
    correct = [float(t == q) for t, q in zip(y_true, y_pred)]
    confs = [r["conf"] if isinstance(r["conf"], (int, float)) else 0.5 for r in results]
    fails = sum(1 for r in results if r["pred"] not in keys)
    lats = [r["latency"] for r in results if r["latency"] > 0]
    resolved = next((r["resolved_model"] for r in results if r["resolved_model"]), args.model_id)

    summary = {
        "n": len(items), "accuracy": round(sum(correct) / len(correct), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, len(keys)), 4),
        "ece": round(ece_score(confs, correct), 4),
        "api_fail": fails,
        "mean_latency_s": round(sum(lats) / len(lats), 2) if lats else 0,
        "total_tokens": sum(r["tokens"] for r in results),
        "total_cost_usd": round(sum(r["cost"] for r in results), 6),
        "model": args.model_id, "resolved_model": resolved,
    }
    out_json = os.path.join(ROOT, "data", "processed", f"eval_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({args.set: summary,
                   "_summary": {"model": resolved, "tag": args.tag}}, f, indent=2)
    dump = os.path.join(ROOT, "data", "processed", f"bench_{args.tag}_items.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        for it, r in zip(items, results):
            f.write(json.dumps({"text": it["state"], "gold": keys[it["label"]],
                                "pred": r["pred"], "conf": r["conf"]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"tersimpan: {out_json} + {dump}")


if __name__ == "__main__":
    main()
