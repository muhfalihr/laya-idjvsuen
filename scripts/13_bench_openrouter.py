"""Benchmark model OpenRouter (mis. ~typesafe/jev-latest) pada test set laya.

Prompt klasifikasi identik semantik dengan format pertanyaan laya (instructions +
opsi "key: description"), jawaban dipaksa satu key opsi. Metrik: accuracy, macro-F1,
parse-failure rate, latensi. Hasih disimpan format eval_* agar bisa dibandingkan
dengan 06/14.

Prasyarat: env OPENROUTER_API_KEY.

Penggunaan:
  python scripts/13_bench_openrouter.py --model-id "~typesafe/jev-latest" \
      --test-sets data/processed/test_sets_tickets.pt --set ticket_intent --n 300 --tag jev
"""
import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def build_prompt(state, instructions, criteria):
    opts = "\n".join(f"- {k}: {d}" for k, d in criteria.items())
    return (
        f"{instructions}\n\n"
        f"Options:\n{opts}\n\n"
        f'Message: """{state}"""\n\n'
        "Which single option fits the message best? "
        "Reply with ONLY the option key, nothing else."
    )


def parse_key(text, keys):
    t = text.strip().strip("`\"'.").lower()
    if t in keys:
        return t
    # cocokkan key yang muncul sebagai kata utuh di jawaban
    for k in keys:
        if re.search(r"\b%s\b" % re.escape(k), t):
            return k
    return None


def call_one(session, model_id, prompt, keys, max_retries=3):
    import requests

    headers = {
        "Authorization": "Bearer %s" % os.environ["OPENROUTER_API_KEY"],
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/muhfalihr/laya-idjvsuen",
        "X-Title": "laya-idjvsuen benchmark",
    }
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = session.post(API_URL, headers=headers, timeout=120, json={
                "model": model_id, "messages": messages, "temperature": 0,
                "max_tokens": 512,
            })
            dt = time.time() - t0
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"] or ""
            key = parse_key(out, keys)
            if key is None and attempt < max_retries - 1:
                messages = messages + [
                    {"role": "assistant", "content": out},
                    {"role": "user", "content": "Invalid. Reply with ONLY one option key from the list."},
                ]
                continue
            usage = r.json().get("usage", {})
            return {"pred": key, "raw": out[:80], "latency": round(dt, 2),
                    "tokens": usage.get("total_tokens")}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"pred": None, "raw": "ERROR %s" % str(e)[:60], "latency": 0, "tokens": 0}
            time.sleep(3 * (attempt + 1))
    return {"pred": None, "raw": "unparsed", "latency": 0, "tokens": 0}


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
    criteria = ts["criteria"]
    instr = ts["instructions"]
    print(f"{len(items)} sampel | {len(keys)} opsi | model {args.model_id}")

    session = requests.Session()
    prompts = [build_prompt(it["state"], instr, criteria) for it in items]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(lambda pr: call_one(session, args.model_id, pr, keys), prompts))

    y_true = [it["label"] for it in items]
    y_pred = [keys.index(r["pred"]) if r["pred"] else -1 for r in results]
    correct = [float(t == q) for t, q in zip(y_true, y_pred)]
    acc = sum(correct) / len(correct)
    parse_fail = sum(1 for r in results if r["pred"] is None)
    lats = [r["latency"] for r in results if r["latency"] > 0]
    toks = sum(r["tokens"] or 0 for r in results)

    summary = {
        "n": len(items), "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1(y_true, y_pred, len(keys)), 4),
        "parse_fail": parse_fail,
        "mean_latency_s": round(sum(lats) / len(lats), 2) if lats else 0,
        "total_tokens": toks,
        "per_label": dict(Counter(keys[i] for i in y_true)),
        "model": args.model_id,
    }
    out_json = os.path.join(ROOT, "data", "processed", f"eval_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({args.set: summary, "_summary": {"model": args.model_id, "tag": args.tag}}, f, indent=2)
    dump = os.path.join(ROOT, "data", "processed", f"bench_{args.tag}_items.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        for it, r in zip(items, results):
            f.write(json.dumps({"text": it["state"], "gold": keys[it["label"]],
                                "pred": r["pred"], "raw": r["raw"]}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"tersimpan: {out_json} + {dump}")


if __name__ == "__main__":
    main()
