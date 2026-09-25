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
  - ticket-classification
library_name: transformers
metrics:
  - accuracy
  - f1
---

# laya-idjvsuen-v3

**Bahasa / Language**: **Indonesia** | [English](MODEL-CARD-V3.en.md)

**Fine-tune domain dari [laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1)**
(yang merupakan fine-tune multibahasa dari
[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)):
decision model non-autoregressive (mmBERT-base, 322M parameter) untuk keputusan terkalibrasi
pada input **Bahasa Indonesia (id), Jawa (jv), Sunda (su), Inggris (en), dan campurannya** —
diperluas dengan tugas **intent 12 kategori domain tiket internal**.

| Model | Fokus | Tautan |
|---|---|---|
| laya-idjvsuen-v1 | multibahasa umum (intent MASSIVE 60-kelas + sentimen NusaX) | [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) |
| **laya-idjvsuen-v3** | v1 **+ 12 kategori tiket** (disarankan untuk routing tiket) | repo ini |

**Model ini tidak menghasilkan teks** — ia menjawab pertanyaan bertipe yang didefinisikan
pemanggil (choice/score/noul) dengan probabilitas terkalibrasi dalam satu forward pass.
Dibangun di atas karya [ConvAI Innovations](https://huggingface.co/convaiinnovations) dengan
SDK open-source [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).

**Penulis**: muhfalihr ([github.com/muhfalihr](https://github.com/muhfalihr))

## 12 kategori domain tiket

Asset & Devices · Infrastructure · Platforms · Employee Support · Security · Compliance ·
Account & Identity · Data & Reporting · Networks & Connectivity · Application Issue ·
Service · Request Access

## Penggunaan

```bash
pip install laya
```

```python
import laya, json, urllib.request

agent = laya.load("faall7479/laya-idjvsuen-v3")

# definisi pertanyaan persis seperti training (12 kategori, urutan opsi benar)
qd = json.load(urllib.request.urlopen(
    "https://huggingface.co/faall7479/laya-idjvsuen-v3/raw/main/question_defs.json"))
q = {"k": {"type": "choice",
           "instructions": qd["ticket_intent"]["instructions"],
           "criteria": qd["ticket_intent"]["criteria"]}}

r = agent.predict("mas tolong buatkan bucket s3 baru untuk data riset", q)
# r["answers"]["k"]["choice"] -> "infrastructure" | probabilities | answer_confidence
```

Kemampuan umum v1 (intent MASSIVE 60-kelas, sentimen NusaX 3-kelas) tetap ada —
lihat card v1 untuk definisi pertanyaan tersebut.

## Data pelatihan

| Task | Bahasa | Train | Sumber | Lisensi |
|---|---|---|---|---|
| Intent MASSIVE (60 kelas) | id, en, jv, su | 13.547+13.547 id/en; 7.000 jv/su (MT NLLB) | MASSIVE 1.1 + NLLB | CC-BY-4.0 / lihat card v1 |
| Sentimen NusaX (3 kelas) | id, jv, su, en | 500/bahasa | anotasi manual penutur asli | CC-BY-SA-4.0 |
| **Intent tiket (12 kelas)** | id (santai + campuran) | 5.731 (termasuk oversample kelas langka ×3) | tiket support internal nyata, **weak label kata kunci** + mining kelas langka rileks | internal |

Total 17.731 item; replay v1 (12.000 item) mencegah forgetting; 400 item hold-out untuk
kalibrasi temperatur. Dilatih dari bobot laya-idjvsuen-v1 dengan resep RLCD resmi Laya
(2 epoch, 1 GPU 8 GB, ±30 menit); temperatur ter-fit: choice 3,44.

## Hasil evaluasi

Test tiket 12-kategori (1.126 pesan nyata, weak label strict):

- Akurasi keseluruhan **94,0%**, macro-F1 0,782, ECE 0,019.
- Kenaikan per-kelas vs v2 (sebelum boost): platforms 0,36→0,64, employee_support 0,75→1,00,
  data_reporting 0,80→1,00, networks 0,58→0,79; rata-rata macro per-kelas 0,830→0,886.

### Benchmark vs Jev (OpenRouter) — 300 sampel nyata identik

![Benchmark keseluruhan](assets/bench_overall.png)

![Matriks per-kategori](assets/bench_perclass.png)

v3 mencapai **93,3% akurasi / ECE 0,015** vs Jev 1.13 sebesar 84,7% — berjalan lokal tanpa
biaya API. Metodologi & catatan jujur: `BENCHMARK.md` di
[repo pipeline](https://github.com/muhfalihr/laya-idjvsuen).

Regresi multibahasa (15 test set vs v1): intent id/en/jv/su dalam 1,6 poin, sentimen Sunda
+3,8 poin, code-switch stabil — replay mencegah forgetting.

## Keterbatasan

1. **Label tiket adalah weak labels** (aturan kata kunci, bukan anotasi manusia) — sebagian
   akurasi adalah kesesuaian dengan "annotator aturan". Audit manual sampel disarankan
   sebelum SLA produksi.
2. **Security & compliance kekurangan data** (21 dan 111 sampel training): phishing tertangkap,
   insiden yang lebih halus bisa jatuh ke `account_identity`. Mengumpulkan ±50-100 kasus
   historis nyata per kelas adalah langkah berikutnya paling bernilai.
3. Data intent jv/su MASSIVE adalah terjemahan mesin (kaveat NC NLLB — lihat card v1);
   angka jujur teks-manusia adalah baris NusaX.
4. Bukan model generatif; kalibrasi di-fit pada distribusi task ini.

## Lisensi & atribusi

- Bobot hasil fine-tune: **CC-BY-SA-4.0** (warisan share-alike NusaX; kaveat NC NLLB pada
  subset jv/su seperti didokumentasikan di card v1).
- Basis: [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0) © ConvAI Innovations; SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).
- MASSIVE 1.1 © Amazon CC-BY-4.0; NusaX-senti © IndoNLP CC-BY-SA-4.0.

## Sitasi

```bibtex
@misc{laya-idjvsuen-v3,
  title  = {laya-idjvsuen-v3: domain fine-tune of the multilingual Laya decision model for Indonesian, Javanese, Sundanese, English, code-switching, and internal ticket routing},
  author = {muhfalihr},
  year   = {2026},
  note   = {Fine-tune of faall7479/laya-idjvsuen-v1 (Apache-2.0 lineage) on MASSIVE 1.1, NusaX-senti, NLLB translations, and real internal ticket data},
  url    = {https://huggingface.co/faall7479/laya-idjvsuen-v3}
}
```
