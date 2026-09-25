# Hasil Retraining Laya — Baseline vs Fine-tuned

**Bahasa**: **Indonesia** | [English](HASIL-RETRAINING.en.md)

- Baseline: `convaiinnovations/laya-multilingual`
- Fine-tuned: `faall7479/laya-idjvsuen-v1`

| Test set | n | Akurasi base | Akurasi FT | Δ | ECE base | ECE FT |
|---|---|---|---|---|---|---|
| Intent Indonesia (MASSIVE 60-kelas) | 2000 | 0.410 | 0.862 | +0.452 | 0.213 | 0.064 |
| Intent Inggris (MASSIVE 60-kelas) | 2000 | 0.479 | 0.857 | +0.378 | 0.203 | 0.056 |
| Intent Jawa (terj. NLLB) | 2000 | 0.281 | 0.805 | +0.524 | 0.251 | 0.081 |
| Intent Sunda (terj. NLLB) | 2000 | 0.252 | 0.772 | +0.520 | 0.211 | 0.080 |
| Sentimen Indonesia (NusaX) | 400 | 0.710 | 0.860 | +0.150 | 0.159 | 0.121 |
| Sentimen Inggris (NusaX) | 400 | 0.743 | 0.873 | +0.130 | 0.121 | 0.105 |
| Sentimen Jawa (NusaX) | 400 | 0.590 | 0.818 | +0.228 | 0.236 | 0.151 |
| Sentimen Sunda (NusaX) | 400 | 0.415 | 0.757 | +0.342 | 0.385 | 0.213 |
| Code-switch id+en (sintetis) | 500 | 0.376 | 0.820 | +0.444 | 0.267 | 0.083 |
| Code-switch id+jv (sintetis) | 500 | 0.350 | 0.816 | +0.466 | 0.277 | 0.071 |
| Code-switch id+sun (sintetis) | 500 | 0.342 | 0.852 | +0.510 | 0.234 | 0.065 |

## Rata-rata per tugas

| Tugas | Akurasi base | Akurasi FT | Δ |
|---|---|---|---|
| Intent (4 bahasa) | 0.355 | 0.824 | +0.468 |
| Sentimen (4 bahasa) | 0.614 | 0.827 | +0.213 |
| Code-switch | 0.356 | 0.829 | +0.473 |
