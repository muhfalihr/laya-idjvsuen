# laya-idjvsuen

**Bahasa**: **Indonesia** | [English](README.en.md)

Fine-tune dari [ConvAI Innovations Laya](https://huggingface.co/convaiinnovations/laya-multilingual)
(*decision model* non-generatif, Apache-2.0) agar kemampuan klasifikasi/keputusannya kuat
untuk input **Bahasa Indonesia, Jawa, Sunda, Inggris, dan campurannya (code-switching)**.

> Laya adalah model keputusan (choice/score/noul dengan probabilitas terkalibrasi),
> bukan model generatif — ia tidak menghasilkan teks. Proyek ini tidak mengubah sifat itu.
> SDK & kode upstream: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (tidak
> disertakan di repo ini, `pip install laya`).

## Versi model

| Versi | Fokus | HuggingFace | Lokal |
|---|---|---|---|
| **v1** | multibahasa umum: intent MASSIVE 60-kelas + sentimen NusaX (id/jv/su/en + CS) | [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) | `runs/laya-idjvsuen-v1` |
| v2 | v1 + domain tiket versi awal (digantikan v3; tidak dipublish) | — | `runs/laya-idjvsuen-v2` |
| **v3** | v1 + domain tiket 12 kategori (kelas langka dikuatkan) — **disarankan untuk routing tiket** | [faall7479/laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3) | `runs/laya-idjvsuen-v3` |
| **v4** | generalisasi dari v1: + topik SIB-200 (id/jv/su/en) + IndoNLU, replay penuh tanpa regresi — **disarankan untuk klasifikasi umum** | [faall7479/laya-idjvsuen-v4](https://huggingface.co/faall7479/laya-idjvsuen-v4) | `runs/laya-idjvsuen-v4` |

*Konvensi: setiap ada versi model baru, tabel ini dan dokumentasi di `docs/` diperbarui.*

## Hasil (akurasi, baseline → fine-tuned)

| Tugas | Baseline | Hasil |
|---|---|---|
| Intent 60-kelas — id / en (MASSIVE) | 41,0% / 47,9% | **86,2% / 85,7%** |
| Intent 60-kelas — jv / su (terj. NLLB) | 28,1% / 25,2% | **80,5% / 77,2%** |
| Sentimen — id / jv / su / en (NusaX) | 41–74% | **76–87%** |
| Code-switch id+en / id+jv / id+su (sintetis) | 34–38% | **82–85%** |

ECE (kalibrasi) turun di semua test set. Model: [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1).

## Benchmark vs Jev (OpenRouter)

Klasifikasi 12 kategori domain tiket internal, **300 sampel nyata identik** untuk semua peserta
(gold = weak labels; metodologi: [docs/BENCHMARK.md](docs/BENCHMARK.md)):

![Benchmark keseluruhan](docs/assets/bench_overall.png)

![Matriks per-kategori](docs/assets/bench_perclass.png)

![Kemampuan multibahasa sebelum vs sesudah fine-tune](docs/assets/bench_languages.png)

Varian terbaru `laya-idjvsuen-v3` (fine-tune domain + penguatan kelas langka) mencapai
**93,3% akurasi / ECE 0,015** pada sampel identik — mengungguli Jev 1.13 (84,7%) dan
berjalan lokal tanpa biaya API.

## Penggunaan

```bash
pip install laya
```

```python
import laya
agent = laya.load("faall7479/laya-idjvsuen-v1")  # atau "faall7479/laya-idjvsuen-v3" (domain tiket)
r = agent.predict("Pelayanane elek tenan, aku ora arep balik maneh", {
    "s": {"type": "choice", "instructions": "What is the sentiment of the text?",
          "criteria": {"negative": "negative opinion",
                       "neutral": "neutral or factual",
                       "positive": "positive opinion"}},
})
print(r["answers"]["s"]["choice"], r["answers"]["s"]["answer_confidence"])
```

## Pipeline (repro)

```bash
python scripts/01_download_data.py     # MASSIVE id/en + NusaX-senti (4 bahasa)
python scripts/02_translate_jav_sun.py # augmentasi jv/su via NLLB (opsional)
python scripts/03_build_dataset.py     # items typed-decisions
python scripts/05_eval.py --tag baseline
python scripts/04_train.py             # RLCD + CE, ~2 jam di GPU 8 GB
python scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned
python scripts/06_compare.py           # tabel baseline vs hasil
python scripts/07_publish_hf.py --repo-id faall7479/laya-idjvsuen-v3  # publikasi (contoh konkret)
```

Tahap 2 (opsional) — adaptasi domain tiket internal (skrip 08–12, data dari database
internal via `secrets/db.ini`): lihat [docs/PIPELINE.md](docs/PIPELINE.md).

Tahap 3 (opsional) — generalisasi **v4** dari v1 di luar domain tiket (SIB-200 topik
id/jv/su/en + IndoNLU + replay): `python scripts/15_build_general.py`, lalu training di
Colab T4 via [notebooks/colab_v4_general.ipynb](notebooks/colab_v4_general.ipynb).

## Dokumentasi lengkap

Semua detail ada di [`docs/`](docs/), tersedia dalam dua bahasa (id + en):

| Dokumen | Indonesia | English |
|---|---|---|
| Riset mendalam | [RISET-LAYA.md](docs/RISET-LAYA.md) | [RISET-LAYA.en.md](docs/RISET-LAYA.en.md) |
| Riset Jev (TypeSafe) + analisis gap | [RISET-JEV.md](docs/RISET-JEV.md) | [RISET-JEV.en.md](docs/RISET-JEV.en.md) |
| Dokumentasi pipeline | [PIPELINE.md](docs/PIPELINE.md) | [PIPELINE.en.md](docs/PIPELINE.en.md) |
| Hasil & evaluasi | [HASIL-RETRAINING.md](docs/HASIL-RETRAINING.md) | [HASIL-RETRAINING.en.md](docs/HASIL-RETRAINING.en.md) |
| Model card HuggingFace | [MODEL-CARD.md](docs/MODEL-CARD.md) | [MODEL-CARD.en.md](docs/MODEL-CARD.en.md) |

## Lisensi

- Kode proyek ini: **Apache-2.0** (mengikuti upstream Laya).
- Bobot model hasil fine-tune: **CC-BY-SA-4.0** — data training NusaX bersifat share-alike;
  lihat [docs/MODEL-CARD.md](docs/MODEL-CARD.md) untuk catatan lisensi penuh
  (termasuk kaveat non-komersial NLLB pada subset intent jv/su).
