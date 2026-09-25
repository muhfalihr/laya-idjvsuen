"""Evaluasi model (baseline laya-multilingual atau hasil fine-tune) pada test set 4 bahasa.

Metrik per test set: accuracy, macro-F1, ECE (answer_confidence vs correctness),
mean confidence, coverage@conf>=0.8. Evaluasi lewat forward langsung + temperatur
dari config checkpoint (identik jalur inference Agent.predict, tanpa router).

Penggunaan:
  python scripts/05_eval.py --model convaiinnovations/laya-multilingual --tag baseline
  python scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convaiinnovations/laya-multilingual")
    p.add_argument("--test-sets", default=os.path.join(ROOT, "data", "processed", "test_sets.pt"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tag", default="eval")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, clamp_temperature
    from hflocal import local_snapshot

    model_dir = args.model if os.path.exists(args.model) else local_snapshot(args.model, ROOT)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    temps = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]

    test_sets = torch.load(args.test_sets, weights_only=False)
    pad_id = tok.pad_token_id
    results = {}
    print(f"{'set':<18}{'n':>6}{'acc':>8}{'macroF1':>9}{'ECE':>8}{'conf':>8}{'cov@0.8':>9}")
    for name, ts in test_sets.items():
        items = ts["items"]
        n_classes = len(ts["label_names"])
        y_true, y_pred, confs = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(items), args.batch_size):
                chunk = items[i:i + args.batch_size]
                L = max(len(it["ids"]) for it in chunk)
                kmax = max(len(it["markers"]) for it in chunk)
                ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                mpos = torch.zeros((len(chunk), kmax), dtype=torch.long)
                mmask = torch.zeros((len(chunk), kmax), dtype=torch.bool)
                for j, it in enumerate(chunk):
                    ids[j, : len(it["ids"])] = torch.tensor(it["ids"])
                    att[j, : len(it["ids"])] = 1
                    mpos[j, : len(it["markers"])] = torch.tensor(it["markers"])
                    mmask[j, : len(it["markers"])] = True
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logits, _ = model(ids.to(device), att.to(device), mpos.to(device),
                                      mmask.to(device),
                                      torch.tensor([it["qtype"] for it in chunk]).to(device))
                logits = logits.float().cpu().numpy()
                for j, it in enumerate(chunk):
                    k = len(it["markers"])
                    z = logits[j, :k] / temps[it["qtype"]]
                    e_z = np.exp(z - z.max())
                    prob = e_z / e_z.sum()
                    y_pred.append(int(prob.argmax()))
                    y_true.append(it["label"])
                    confs.append(float(prob.max()))
        correct = [float(t == q) for t, q in zip(y_true, y_pred)]
        acc = float(np.mean(correct))
        f1 = macro_f1(y_true, y_pred, n_classes)
        ece = ece_score(confs, correct)
        cov = float(np.mean([c >= 0.8 for c in confs])) if confs else 0.0
        results[name] = {"n": len(items), "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                         "ece": round(ece, 4), "mean_conf": round(float(np.mean(confs)), 4),
                         "coverage@0.8": round(cov, 4), "sec": round(time.time() - t0, 1)}
        print(f"{name:<18}{len(items):>6}{acc:>8.3f}{f1:>9.3f}{ece:>8.3f}"
              f"{float(np.mean(confs)):>8.3f}{cov:>9.3f}")

    agg = {}
    for group in ("intent", "sentiment", "cs"):
        sel = [r["accuracy"] for k, r in results.items() if k.startswith(group)]
        if sel:
            agg[group] = round(float(np.mean(sel)), 4)
    results["_summary"] = {"model": args.model, "tag": args.tag, "aggregate_accuracy": agg}

    out_path = os.path.join(ROOT, "data", "processed", f"eval_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nhasil tersimpan: {out_path}")
    print("aggregate:", json.dumps(agg))


if __name__ == "__main__":
    main()
