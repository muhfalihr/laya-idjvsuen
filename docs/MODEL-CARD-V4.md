---
language:
  - id
  - jv
  - su
  - en
license: cc-by-sa-4.0
base_model: faall7479/laya-idjvsuen-v1
tags:
  - text-classification
  - decision-model
  - laya
  - indonesian
  - javanese
  - sundanese
  - code-switching
  - topic-classification
library_name: transformers
metrics:
  - accuracy
  - f1
---

# laya-idjvsuen-v4

**Bahasa / Language**: **Indonesia** | [English](MODEL-CARD-V4.en.md)

**Fine-tune general dari [laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1)**
(yang merupakan fine-tune multibahasa dari
[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)):
decision model non-autoregressive (mmBERT-base, 322M parameter) untuk keputusan terkalibrasi
pada input **Bahasa Indonesia (id), Jawa (jv), Sunda (su), Inggris (en), dan campurannya** —
diperluas ke arah **umum** (bukan domain tiket): klasifikasi **topik 7 kategori** dan tugas
IndoNLU (emosi, sentimen review, aspek), dengan **replay penuh data v1** sehingga semua
kemampuan lama bertahan bahkan meningkat.

| Model | Fokus | Tautan |
|---|---|---|
| laya-idjvsuen-v1 | multibahasa umum (intent MASSIVE 60-kelas + sentimen NusaX) | [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) |
| laya-idjvsuen-v3 | v1 + 12 kategori tiket (untuk routing tiket) | [faall7479/laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3) |
| **laya-idjvsuen-v4** | **generalisasi dari v1**: + topik SIB-200 + IndoNLU, replay penuh — **untuk klasifikasi umum** | repo ini |

**Model ini tidak menghasilkan teks** — ia menjawab pertanyaan bertipe yang didefinisikan
pemanggil (choice/score/noul) dengan probabilitas terkalibrasi dalam satu forward pass.
Dibangun di atas karya [ConvAI Innovations](https://huggingface.co/convaiinnovations) dengan
SDK open-source [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).

**Penulis**: muhfalihr ([github.com/muhfalihr](https://github.com/muhfalihr))

## Penggunaan

```bash
pip install laya
```

```python
import laya, json, urllib.request

agent = laya.load("faall7479/laya-idjvsuen-v4")

# tugas BARU: klasifikasi topik (definisi persis seperti training)
qd = json.load(urllib.request.urlopen(
    "https://huggingface.co/faall7479/laya-idjvsuen-v4/raw/main/question_defs_general.json"))
q = {"t": {"type": "choice",
           "instructions": qd["topic"]["instructions"],
           "criteria": qd["topic"]["criteria"]}}

r = agent.predict("Timnas berjuang di laga pamungkas kualifikasi", q)
# r["answers"]["t"]["choice"] -> "sports" | probabilities | answer_confidence
```

Kemampuan v1 (intent MASSIVE 60-kelas, sentimen NusaX 3-kelas — `question_defs.json` di repo
ini) tetap tersedia dan tidak menurun.

## Data pelatihan

| Task | Bahasa | Train | Sumber | Lisensi |
|---|---|---|---|---|
| Intent MASSIVE (60 kelas) — replay | id, en, jv, su | 13.547+13.547 id/en; 7.000 jv/su (MT NLLB) | MASSIVE 1.1 + NLLB | CC-BY-4.0 / lihat card v1 |
| Sentimen NusaX (3 kelas) — replay | id, jv, su, en | 500/bahasa | anotasi manual penutur asli | CC-BY-SA-4.0 |
| **Topik SIB-200 (7 kelas)** | id, jv, su, en | 800/bahasa (FLORES-200 paralel) | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC-BY |
| **IndoNLU: emosi/sentimen-review/aspek** | id | ±12 ribu item | [indonlp/indonlu](https://huggingface.co/datasets/indonlp/indonlu) (emot, smsa, casa) | bervariasi per sub-dataset |

Code-switch sintetis hanya untuk evaluasi. Dilatih dari bobot laya-idjvsuen-v1 di
**Google Colab T4** (fp16 + GradScaler, adamw8bit, batch efektif 64, 2 epoch ≈ 3 jam);
temperatur ter-fit: choice 3,55.

## Hasil evaluasi (v1 → v4, test set identik)

**Kemampuan baru — topik 7 kategori (SIB-200):**

| Set | v1 | v4 | ECE v1→v4 |
|---|---|---|---|
| Topik — id / en | 78,4% / 77,0% | **87,3% / 89,2%** | 0,157→0,080 / 0,165→0,066 |
| Topik — jv / su | 72,1% / 66,2% | **84,3% / 79,9%** | 0,159→0,104 / 0,126→0,111 |
| Topik code-switch (6 kombinasi) | 71,3–78,4% | **88,2–92,0%** | 0,149–0,185 → 0,046–0,066 |

Coverage@conf≥0,8 topik naik dari ±45% (v1) menjadi ±90% (v4) — layak automasi.

**Kemampuan lama — replay penuh, tanpa regresi (justru naik):**

| Set | v1 | v4 |
|---|---|---|
| Intent id / en | 86,2% / 85,7% | **87,3% / 87,6%** |
| Intent jv / su | 80,5% / 77,2% | **82,3% / 80,0%** |
| Sentimen id / jv / su / en | 86,0% / 81,8% / 75,7% / 87,3% | **94,0% / 86,0% / 82,0% / 89,7%** |
| Code-switch intent (3 kombinasi) | 81,6–85,2% | **83,4–86,4%** |

Lonjakan sentimen id (+8 poin) konsisten dengan transfer dari `smsa` (review berbahasa
Indonesia) di IndoNLU. Metodologi & tabel lengkap 21 test set: `eval_v4-general.json` di
[repo pipeline](https://github.com/muhfalihr/laya-idjvsuen).

## Keterbatasan

1. Kalimat jv/su SIB-200 berasal dari terjemahan FLORES-200 — indikatif, bukan anotasi
   penutur asli (sama seperti kaveat NLLB pada card v1); baris paling jujur untuk teks
   manusia tetap NusaX.
2. Set evaluasi IndoNLU tidak dijalankan ulang lokal (loader lama; training-nya tetap ikut
   di Colab) — efeknya terlihat tidak langsung lewat kenaikan sentimen id.
3. Tidak dilatih pada data tiket — untuk routing tiket internal gunakan
   [laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3).
4. Bukan model generatif; kalibrasi di-fit pada distribusi task ini.

## Lisensi & atribusi

- Bobot hasil fine-tune: **CC-BY-SA-4.0** (warisan share-alike NusaX; kaveat NC NLLB pada
  subset jv/su seperti di card v1).
- Basis: [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) →
  [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0) © ConvAI Innovations; SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).
- MASSIVE 1.1 © Amazon CC-BY-4.0; NusaX-senti © IndoNLP CC-BY-SA-4.0;
  SIB-200 (Adelani dkk., EACL 2024) CC-BY; IndoNLU (Wilie dkk., AACL 2020) lisensi per-subdataset.

## Sitasi

```bibtex
@misc{laya-idjvsuen-v4,
  title  = {laya-idjvsuen-v4: general fine-tune of the multilingual Laya decision model — SIB-200 topics and IndoNLU tasks for Indonesian, Javanese, Sundanese, English, and code-switching},
  author = {muhfalihr},
  year   = {2026},
  note   = {Fine-tune of faall7479/laya-idjvsuen-v1 with full v1 replay (no forgetting) plus SIB-200 topics and IndoNLU; trained on a free Colab T4},
  url    = {https://huggingface.co/faall7479/laya-idjvsuen-v4}
}
```
