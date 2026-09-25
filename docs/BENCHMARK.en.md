# Benchmark: Laya (no fine-tune) vs Jev (OpenRouter) vs laya-idjvsuen

**Bahasa / Language**: [Indonesia](BENCHMARK.md) | **English**

Task: **12-category internal ticket-domain** classification on real support messages
(casual Indonesian + code-mixing). All participants evaluated on the **identical 300
samples**, identical options & instructions; gold = weak labels (keyword rules — not
human annotation).

## Headline results (identical 300 samples)

| Model | Accuracy | Macro-F1 | ECE | Latency | Cost |
|---|---|---|---|---|---|
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) (no fine-tune) | 24.7% | 0.313 | 0.536 | local, ~ms | $0 |
| [`laya-idjvsuen-v1`](https://huggingface.co/faall7479/laya-idjvsuen-v1) (general id/jv/su/en) | 44.7% | 0.354 | 0.117 | local, ~ms | $0 |
| Jev 1.13 (`~typesafe/jev-latest`, OpenRouter) | 84.7% | 0.564 | 0.083 | 0.32 s (API) | $0.0093 |
| `laya-idjvsuen-v2` (domain fine-tune; superseded by v3, not published) | **96.0%** | **0.752** | 0.030 | local, ~ms | $0 |
| [`laya-idjvsuen-v3`](https://huggingface.co/faall7479/laya-idjvsuen-v3) (domain + rare-class boost) | **93.3%** | 0.680 | **0.015** | local, ~ms | $0 |

v2 leads overall accuracy by 2.7 points but is weak on rare classes; v3 is the released
model because it is balanced per-class (macro per-class 0.830 → 0.886, best ECE) — see the
v3 report for details.

Full 1,126-item test: v2 = 95.2% accuracy / 0.787 macro-F1 / ECE 0.024.

## Anti-forgetting regression (v2 vs v1, the 15 old test sets)

Replaying 12,000 old items during v2 training preserved prior capability: intent
id/en/jv/su drops ≤1.6 points, Sundanese sentiment actually gains +3.8 points,
code-switching stable (±1 point). Without replay, degradation is typically far larger.

## Methodology

- **laya participants**: official SDK forward path (checkpoint-config temperatures), no router.
- **Jev**: OpenRouter `/api/alpha/decisions` endpoint (laya-compatible schema:
  state + questions{type, instructions, criteria}); `~typesafe/jev-latest` resolved to
  `typesafe/jev-1.13-20260917`; default API temperature; 0 API failures.
- Scripts: `scripts/13_bench_openrouter.py` (Jev) and `scripts/05_eval.py` (laya);
  result files `data/processed/eval_{base_ticket300,v1_ticket300,jev,v2_ticket300}.json`.

## Honest notes

1. **Gold = weak labels.** Part of v2's edge is "having learned the annotator (the
   rules)". Jev's 84.7% zero-shot against those same labels is genuinely strong for a
   generic model. For human-neutral numbers, a manual audit of a sample (~200-500
   messages) is needed.
2. **Rare classes** (security, compliance, platforms) have almost no training data —
   accuracy on them is low for every participant; macro-F1 exposes this gap.
3. Jev is a generic hosted decision model — this benchmark demonstrates the value of
   domain fine-tuning, not a weakness of Jev as a product.
4. Cost/latency: laya runs locally (even CPU suffices, 322M); Jev needs an API round-trip.
