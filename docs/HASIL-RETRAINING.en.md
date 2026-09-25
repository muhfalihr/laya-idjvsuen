# Laya Retraining Results — Baseline vs Fine-tuned

**Bahasa / Language**: [Indonesia](HASIL-RETRAINING.md) | **English**

- Baseline: `convaiinnovations/laya-multilingual`
- Fine-tuned: `runs/laya-idjvsuen-v1`

| Test set | n | Baseline acc | FT acc | Δ | Baseline ECE | FT ECE |
|---|---|---|---|---|---|---|
| Indonesian intent (MASSIVE 60-class) | 2000 | 0.410 | 0.862 | +0.452 | 0.213 | 0.064 |
| English intent (MASSIVE 60-class) | 2000 | 0.479 | 0.857 | +0.378 | 0.203 | 0.056 |
| Javanese intent (NLLB-translated) | 2000 | 0.281 | 0.805 | +0.524 | 0.251 | 0.081 |
| Sundanese intent (NLLB-translated) | 2000 | 0.252 | 0.772 | +0.520 | 0.211 | 0.080 |
| Indonesian sentiment (NusaX) | 400 | 0.710 | 0.860 | +0.150 | 0.159 | 0.121 |
| English sentiment (NusaX) | 400 | 0.743 | 0.873 | +0.130 | 0.121 | 0.105 |
| Javanese sentiment (NusaX) | 400 | 0.590 | 0.818 | +0.228 | 0.236 | 0.151 |
| Sundanese sentiment (NusaX) | 400 | 0.415 | 0.757 | +0.342 | 0.385 | 0.213 |
| Code-switch id+en (synthetic) | 500 | 0.376 | 0.820 | +0.444 | 0.267 | 0.083 |
| Code-switch id+jv (synthetic) | 500 | 0.350 | 0.816 | +0.466 | 0.277 | 0.071 |
| Code-switch id+su (synthetic) | 500 | 0.342 | 0.852 | +0.510 | 0.234 | 0.065 |

## Per-task averages

| Task | Baseline acc | FT acc | Δ |
|---|---|---|---|
| Intent (4 languages) | 0.355 | 0.824 | +0.468 |
| Sentiment (4 languages) | 0.614 | 0.827 | +0.213 |
| Code-switching | 0.356 | 0.829 | +0.473 |
