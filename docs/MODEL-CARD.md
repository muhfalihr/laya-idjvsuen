---
language:
  - id
  - jv
  - su
  - en
license: cc-by-sa-4.0
base_model: convaiinnovations/laya-multilingual
tags:
  - text-classification
  - decision-model
  - laya
  - indonesian
  - javanese
  - sundanese
  - code-switching
library_name: transformers
metrics:
  - accuracy
  - f1
---

# laya-idjvsuen-v1

**Bahasa / Language**: **Indonesia** | [English](MODEL-CARD.en.md)

**Fine-tune multibahasa dari [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)**
untuk pengambilan keputusan terkalibrasi (*typed decisions*: `choice` / `score` / `noul`)
pada input **Bahasa Indonesia (id), Jawa (jv), Sunda (su), Inggris (en), dan campurannya**.

**Penulis**: Muhammad Falih Romadhoni ([faall7479](https://huggingface.co/faall7479))

Ini adalah *decision model* non-autoregressive berbasis encoder (mmBERT-base, 322M parameter):
satu forward pass menghasilkan jawaban terketik + probabilitas terkalibrasi. **Model ini tidak
menghasilkan teks** dan bukan chatbot. Dibangun di atas karya [ConvAI Innovations](https://huggingface.co/convaiinnovations)
dengan SDK open-source [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).

## Penggunaan

```bash
pip install laya
```

```python
import laya

agent = laya.load("<repo-id-ini>")

# Sentimen (choice, 3 opsi)
r = agent.predict("Pelayanane elek tenan, aku ora arep balik maneh", {
    "sentimen": {"type": "choice",
                 "instructions": "What is the sentiment of the text?",
                 "criteria": {"negative": "the text expresses a negative opinion",
                              "neutral": "the text is neutral or factual",
                              "positive": "the text expresses a positive opinion"}}
})
# r["answers"]["sentimen"]["choice"] -> "negative" | "probabilities" | "answer_confidence"

# Intent (choice, 60 opsi MASSIVE)
opts = {"alarm_set": None, "calendar_set": None, "qa_factoid": None, ...}  # 60 intent MASSIVE
r = agent.predict("bangunkan saya pukul lima pagi minggu ini", {
    "intent": {"type": "choice",
               "instructions": "Classify the user's utterance into the most likely intent.",
               "criteria": opts}
})
```

Untuk daftar lengkap 60 intent beserta urutan opsi persis yang dipakai saat pelatihan,
lihat berkas `question_defs.json` pada repo model ini (atau
[MASSIVE](https://github.com/alexa/massive)).

## Data pelatihan

| Task | Bahasa | Train | Test | Sumber | Lisensi data |
|---|---|---|---|---|---|
| Intent (60 kelas) | id, en | 13.547 / bahasa | 2.000 / bahasa | MASSIVE 1.1 | CC-BY-4.0 |
| Intent (60 kelas) | jv, su | 7.000 / bahasa | 2.000 / bahasa | terjemahan mesin id→jv/su (NLLB-200-distilled-600M) | — lihat catatan |
| Sentimen (3 kelas) | id, jv, su, en | 500 / bahasa | 400 / bahasa | NusaX-senti (anotasi manual penutur asli) | CC-BY-SA-4.0 |
| Code-switch (eval saja) | id+en, id+jv, id+su | — | 500 / kombinasi | sintetis belahan kalimat paralel | — |

Total 43.094 item pelatihan; 400 item di-hold-out untuk fit temperatur kalibrasi.

## Prosedur pelatihan

Resep RLCD dari [notebook resmi Laya](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb),
diadaptasi untuk 1 GPU konsumen:

- Objective: policy gradient (REINFORCE, group-mean baseline, G=4, σ 0,4→0,1) dengan reward
  *strictly proper scoring rules* (log + spherical 0,75 + RPS 1,0) **+** soft cross-entropy penuh.
- Optimizer: AdamW 8-bit, wd 0,01 — LR encoder 2,5e-5, LR head 1e-4, cosine decay + warmup 60 step.
- Batch efektif 64 (4 × 16 accum), bf16 autocast, gradient checkpointing.
- 3 epoch ≈ 103 menit pada 1× RTX 5050 Laptop 8 GB.
- Paskalibrasi: temperatur per tipe pertanyaan di-fit pada hold-out — `choice` 3,52
  (ship di `rl_agent_config.json`, otomatis dipakai SDK).
- Perubahan config vs checkpoint dasar: `max_len` 1024→768, `head_max_len` 256→512
  (agar 60 opsi intent tidak terpotong).

## Hasil evaluasi

Akurasi pada test set (evaluasi lewat jalur inference SDK, temperatur ship):

| Test set | n | Baseline | Model ini | ECE base → FT |
|---|---|---|---|---|
| Intent id (MASSIVE) | 2.000 | 0,410 | **0,862** | 0,213 → 0,064 |
| Intent en (MASSIVE) | 2.000 | 0,479 | **0,857** | 0,203 → 0,056 |
| Intent jv (terj. NLLB) | 2.000 | 0,281 | **0,805** | 0,251 → 0,081 |
| Intent su (terj. NLLB) | 2.000 | 0,252 | **0,772** | 0,211 → 0,080 |
| Sentimen id (NusaX) | 400 | 0,710 | **0,860** | 0,159 → 0,121 |
| Sentimen en (NusaX) | 400 | 0,743 | **0,873** | 0,121 → 0,105 |
| Sentimen jv (NusaX) | 400 | 0,590 | **0,818** | 0,236 → 0,151 |
| Sentimen su (NusaX) | 400 | 0,415 | **0,757** | 0,385 → 0,213 |
| CS id+en (sintetis) | 500 | 0,376 | **0,820** | 0,267 → 0,083 |
| CS id+jv (sintetis) | 500 | 0,350 | **0,816** | 0,277 → 0,071 |
| CS id+su (sintetis) | 500 | 0,342 | **0,852** | 0,234 → 0,065 |

Macro-F1 dan detail lengkap: [`HASIL-RETRAINING.md`](https://github.com/<user>/laya-idjvsuen/blob/main/docs/HASIL-RETRAINING.md).

## Keterbatasan

1. **Bukan model generatif** — hanya menjawab pertanyaan bertipe (choice/score/noul) yang
   didefinisikan pemanggil, dengan opsi yang ditentukan pemanggil.
2. **Intent jv/su dilatih & diuji pada teks terjemahan mesin** (NLLB). Akurasi jv/su pada teks
   tulisan manusia asli diperkirakan lebih rendah; angka jujur teks-manusia adalah baris NusaX.
3. **Evaluasi code-switching bersifat sintetis** (belahan kalimat paralel), bukan CS alami
   media sosial. Benchmark CS EN-ID publik yang matang belum ada.
4. **Cakupan task terbatas**: intent MASSIVE + sentimen. Kemampuan typed-decisions pada task
   lain (mis. benchmark asli Laya: Banking77, SST-5, XNLI) belum diukur ulang pasca-fine-tune.
5. Kalibrasi di-fit pada distribusi dua task ini; gunakan `answer_confidence` dengan hati-hati
   di luar distribusinya.

## Lisensi & atribusi

- Bobot hasil fine-tune ini dirilis **CC-BY-SA-4.0** (data NusaX bersifat share-alike).
- Model dasar [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual):
  **Apache-2.0** © ConvAI Innovations; SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya): Apache-2.0.
- MASSIVE 1.1 © Amazon: CC-BY-4.0. NusaX-senti © IndoNLP: CC-BY-SA-4.0.
- ⚠️ Subset intent jv/su dihasilkan dengan model [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
  yang berlisensi **CC-BY-NC-4.0** (non-komersial). Untuk penggunaan komersial, disarankan
  mengganti subset ini dengan terjemahan dari model MT berlisensi terbuka atau anotasi manusia
  lalu melatih ulang (pipeline lengkap tersedia di repo).

## Mengutip

```bibtex
@misc{laya-idjvsuen-v1,
  title  = {laya-idjvsuen-v1: multilingual Laya decision model fine-tuned for Indonesian, Javanese, Sundanese, English, and code-switching},
  author = {Muhammad Falih Romadhoni},
  year   = {2026},
  note   = {Fine-tune of convaiinnovations/laya-multilingual (Apache-2.0) on MASSIVE 1.1, NusaX-senti, and NLLB-generated translations},
  url    = {https://huggingface.co/<repo-id-ini>}
}
```
