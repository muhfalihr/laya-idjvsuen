# Jalur A — Fine-tune Laya Multilingual untuk Bahasa Indonesia, Jawa, Sunda, Inggris

**Bahasa**: **Indonesia** | [English](PIPELINE.en.md)

Proyek ini me-retain **`laya-multilingual`** (decision model non-generatif, 322M, Apache-2.0)
agar kemampuan *decision/classification*-nya kuat untuk input bahasa Indonesia (`id`),
Jawa (`jv`), Sunda (`sun`), dan Inggris (`en`). Laporan riset lengkap: [RISET-LAYA.md](RISET-LAYA.md).

## ✅ Hasil (24 Sep 2026, model: [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1), lokal `runs/laya-idjvsuen-v1`)

Tabel lengkap: [HASIL-RETRAINING.md](HASIL-RETRAINING.md). Ringkasan akurasi:

| Tugas | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Intent 60-kelas (rata-rata 4 bahasa) | 35,5% | **82,4%** | +46,8 |
| — Indonesia / Inggris | 41,0% / 47,9% | **86,2% / 85,7%** | +45 / +38 |
| — Jawa / Sunda (terj. NLLB) | 28,1% / 25,2% | **80,5% / 77,2%** | +52 / +52 |
| Sentimen (rata-rata 4 bahasa) | 61,4% | **82,7%** | +21,3 |
| — Sunda (NusaX, teks asli manusia) | 41,5% | **75,7%** | +34,2 |
| Code-switch (rata-rata) | 35,6% | **82,9%** | +47,3 |

ECE turun di semua set (mis. intent id 0,213→0,064); kalibrasi temperatur per-tipe ter-fit.
Catatan jujur: test intent jv/su & CS memakai teks terjemahan/sintetis (bukan ground-truth
manusia); angka jujur untuk teks asli manusia adalah baris NusaX (sentimen). Kemampuan
typed-decisions di luar dua task ini belum diukur ulang (potensi forgetting — lihat TODO).


## Setup

```bash
# venv sudah ada di .venv (Python 3.13, torch 2.11 cu128)
.venv/Scripts/python.exe -m pip install -r requirements.txt   # bila perlu ulang
```

## Pipeline

| Langkah | Skrip | Output | Waktu (RTX 5050 8GB) |
|---|---|---|---|
| 1. Download data mentah | `scripts/01_download_data.py` | `data/raw/` (MASSIVE id+en, NusaX-senti 4 bahasa) | ~5-10 menit |
| 2. (Opsional) Terjemahkan id→jv/sun | `scripts/02_translate_jav_sun.py` | `data/translated/massive-{jv,sun}.jsonl` (NLLB-600M) | ~20-40 menit |
| 3. Bangun dataset typed-decisions | `scripts/03_build_dataset.py` | `data/processed/{train_items,test_sets}.pt` | ~2-5 menit |
| 4. Evaluasi baseline | `scripts/05_eval.py --tag baseline` | `data/processed/eval_baseline.json` | ~5-15 menit |
| 5. Training | `scripts/04_train.py` | `runs/laya-idjvsuen-v1/` | ~2 jam (3 epoch) |
| 6. Evaluasi hasil | `scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned` | `eval_finetuned.json` | ~5-15 menit |
| 7. Bandingkan | `scripts/06_compare.py` | `docs/HASIL-RETRAINING{,.en}.md` | detik |
| 8. Publikasi HF | `scripts/07_publish_hf.py --repo-id <user>/...` | repo HuggingFace | menit |

## Tahap 2 (opsional) — adaptasi domain tiket internal

Adaptasi lanjutan pada data percakapan tiket dari dua database internal
(PostgreSQL + MySQL; kredensial di `secrets/db.ini` yang di-gitignore):

| Langkah | Skrip | Fungsi |
|---|---|---|
| 9. Ambil data | `scripts/08_fetch_tickets.py` | tarik tabel pesan tiket kedua DB → JSONL mentah |
| 10. Preprocessing | `scripts/09_preprocess_tickets.py` | normalisasi, masking PII/secret, dedup, split user/handler, statistik bahasa |
| 11. Evaluasi produksi | `scripts/10_eval_tickets.py` | uji model pada sampel pesan nyata |
| 12. Weak labeling | `scripts/11_weak_label_tickets.py` | label otomatis 12 kategori domain (aturan kata kunci id+en) |
| 13. Dataset v2 | `scripts/12_build_ticket_intent.py` | item 12-kategori + split per ticket + replay data v1 |
| 14. Training v2 | `scripts/04_train.py --model runs/laya-idjvsuen-v1 --data data/processed/train_items_v2.pt --out runs/laya-idjvsuen-v2 --model-name ...` | ~30 menit |

Kategori domain (12): Asset & Devices, Infrastructure, Platforms, Employee Support,
Security, Compliance, Account & Identity, Data & Reporting, Networks & Connectivity,
Application Issue, Service, Request Access. Catatan jujur: label tahap ini adalah
*weak labels* (kata kunci, bukan anotasi manusia) — metriknya "weak-label accuracy".

### Menjalankan

```bash
cd D:\MyTools\laya
.venv/Scripts/python.exe scripts/01_download_data.py
.venv/Scripts/python.exe scripts/02_translate_jav_sun.py      # augmentasi jv/sun (opsional tapi disarankan)
.venv/Scripts/python.exe scripts/03_build_dataset.py
.venv/Scripts/python.exe scripts/05_eval.py --tag baseline    # angka sebelum training
.venv/Scripts/python.exe scripts/04_train.py                  # --resume untuk lanjut; --optim adamw8bit bila OOM
.venv/Scripts/python.exe scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned
```

## Desain

**Data** (semua berlabel, sumber primer):

| Task | Bahasa | Train | Test | Sumber |
|---|---|---|---|---|
| Intent (60 kelas) | id | 14.507 (train+dev) | ≤2000 | MASSIVE 1.1, CC BY 4.0 |
| Intent | en | 14.507 | ≤2000 | MASSIVE 1.1 |
| Intent | jv, sun | 6.000/bahasa | ≤2000/bahasa | NLLB terjemahan id (sintetis) |
| Sentimen (3 kelas) | id, jv, sun, en | 500/bahasa | 400/bahasa | NusaX-senti (manual, CC-BY-SA 4.0) |
| Code-switch (eval saja) | id+en, id+jv, id+sun | — | 500/kombinasi | sintetis belahan kalimat paralel |

**Training** — resep resmi notebook `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`,
diadaptasi 1 GPU: RLCD (policy gradient + proper scoring rules, group baseline) + supervised
soft cross-entropy, LR encoder 2.5e-5 / head 1e-4, cosine decay + warmup, bf16 autocast,
gradient checkpointing (encoder + head), effective batch 64, 3 epoch, lalu **fit temperatur
kalibrasi per tipe pertanyaan** pada 400 item hold-out.

**Perubahan config vs checkpoint asli**: `max_len` 1024→768, `head_max_len` 256→512
(60 opsi intent tidak lagi terpotong menjadi awalan 3 token yang sama). Config hasil training
menuliskan nilai ini sehingga evaluasi via `laya.load()` memakai budget yang sama.

**Evaluasi** — accuracy, macro-F1, ECE, mean confidence, coverage@0.8 per (task, bahasa);
forward path identik jalur inference `Agent.predict` (temperatur dari config), tanpa Router
(load checkpoint multilingual langsung — menghindari bug routing PR #286).

## Inferensi dengan model hasil training

```python
import laya
agent = laya.load(r"D:\MyTools\laya\runs\laya-idjvsuen-v1")
r = agent.predict("Waktune sindur enak banget", {
  "sentimen": {"type": "choice", "instructions": "What is the sentiment of the text?",
               "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}
})
```

## Catatan penting

- **Data jv/su intent adalah terjemahan mesin** (NLLB) — bagus untuk adaptasi bahasa, tapi
  angka evaluasi intent jv/su adalah indikatif, bukan ground-truth manusia. Metrik utama
  tetap test set asli id/en + NusaX-senti (manual penutur asli).
- **Lisensi**: model Apache-2.0; MASSIVE CC BY 4.0; NusaX **CC-BY-SA 4.0 (share-alike)** —
  model yang dilatih dengannya wajib dibagikan dengan lisensi serupa bila didistribusikan.
- **Fallback Kaggle** bila GPU lokal kurang: notebook resmi 2×T4 — upload `train_items.pt`
  hasil langkah 3, ganti MODEL_ID menjadi path multilingual (`convaiinnovations/laya-multilingual`,
  encoder `jhu-clsp/mmBERT-base`).
- VRAM: full fine-tune 322M ≈ 6-6,5 GB (fp32 master + AdamW + aktivasi bf16 grad-ckpt).
  Bila OOM: `--micro-batch 4` atau `--optim adamw8bit`.

## Tahap 3 (opsional) — Generalisasi v4: fine-tune dari v1 di luar domain tiket

Memperluas kemampuan **umum** model (bukan menggantikan v3 untuk routing tiket), tetap
berbasis replay agar kemampuan MASSIVE/NusaX/code-switch v1 tidak dilupakan:

- **SIB-200** — topik 7 kategori, paralel `id`/`jv`/`sun`/`eng` (HF `Davlan/sib200`,
  config `ind_Latn`/`jav_Latn`/`sun_Latn`/`eng_Latn`, ~800 train + 204 test per bahasa, CC-BY)
- **IndoNLU** (opsional otomatis) — `emot`/`smsa`/`casa` bahasa Indonesia; loader lama
  butuh `datasets<3.0`, bila tidak kompatibel dilewati lembut dengan peringatan
- **CS sintetis** — pasangan kalimat SIB se-kategori dicampur setengah-setengah (eval saja)
- **Replay v1** — seluruh `train_items.pt` (43k item) ikut dilatih ulang

```bash
python scripts/15_build_general.py --replay data/processed/train_items.pt
# output: train_items_general.pt + test_sets_general.pt (set baru)
#         + test_sets_v4.pt (gabungan 15 set lama + set baru, untuk 05_eval satu jalan)

python scripts/04_train.py --model <dir-checkpoint-v1> \
    --data data/processed/train_items_general.pt \
    --out runs/laya-idjvsuen-v4 --model-name laya-idjvsuen-v4-general \
    --amp fp16 --optim adamw8bit --micro-batch 8 --grad-accum 8 --epochs 2

python scripts/05_eval.py --model runs/laya-idjvsuen-v4 \
    --test-sets data/processed/test_sets_v4.pt --amp fp16 --tag v4-general
```

**v4 TERPUBLIKASI 2026-09-26**: [faall7479/laya-idjvsuen-v4](https://huggingface.co/faall7479/laya-idjvsuen-v4) —
topik SIB-200 id 87,3% / jv 84,3% / su 79,9% / en 89,2% (v1: 66–78%), CS topik 88–92%,
dan seluruh kemampuan v1 naik tanpa regresi (sentimen id +8 poin via transfer IndoNLU).
Hasil lengkap: [results/eval_v4-general.json](results/eval_v4-general.json) vs
[results/eval_v1-on-v4sets.json](results/eval_v1-on-v4sets.json); card: [MODEL-CARD-V4.md](MODEL-CARD-V4.md).

Jalur **Google Colab T4** siap-jalan (fp16 + GradScaler — T4 tidak punya bf16; checkpoint
per epoch ke Google Drive; `--resume` aman dari disconnect):
[`notebooks/colab_v4_general.ipynb`](../notebooks/colab_v4_general.ipynb).

Alternatif data bila ingin volume lebih besar atau tanpa SIB-200: lihat menu sumber di
[RISET-JEV.md](RISET-JEV.md) bagian 8 (Open-Jev untuk sisi en-general — legal karena
dataset publik, bukan output API; NusaCrowd untuk tugas bahasa daerah lain).
