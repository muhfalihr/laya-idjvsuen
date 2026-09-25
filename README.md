# laya-idjvsuen

Fine-tune dari [ConvAI Innovations Laya](https://huggingface.co/convaiinnovations/laya-multilingual)
(*decision model* non-generatif, Apache-2.0) agar kemampuan klasifikasi/keputusannya kuat
untuk input **Bahasa Indonesia, Jawa, Sunda, Inggris, dan campurannya (code-switching)**.

> Laya adalah model keputusan (choice/score/noul dengan probabilitas terkalibrasi),
> bukan model generatif — ia tidak menghasilkan teks. Proyek ini tidak mengubah sifat itu.
> SDK & kode upstream: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (tidak
> disertakan di repo ini, `pip install laya`).

## Hasil (akurasi, baseline → fine-tuned)

| Tugas | Baseline | Hasil |
|---|---|---|
| Intent 60-kelas — id / en (MASSIVE) | 41,0% / 47,9% | **86,2% / 85,7%** |
| Intent 60-kelas — jv / su (terj. NLLB) | 28,1% / 25,2% | **80,5% / 77,2%** |
| Sentimen — id / jv / su / en (NusaX) | 41–74% | **76–87%** |
| Code-switch id+en / id+jv / id+su (sintetis) | 34–38% | **82–85%** |

ECE (kalibrasi) turun di semua test set. Model tersimpan di `runs/laya-idjvsuen-v1`.

## Penggunaan

```bash
pip install laya
```

```python
import laya
agent = laya.load("runs/laya-idjvsuen-v1")  # atau "<username-hf>/laya-idjvsuen-v1"
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
python scripts/07_publish_hf.py --repo-id <user>/laya-idjvsuen-v1  # publikasi
```

## Dokumentasi lengkap

Semua detail ada di [`docs/`](docs/): [riset mendalam](docs/RISET-LAYA.md) ·
[dokumentasi pipeline](docs/PIPELINE.md) · [hasil & evaluasi](docs/HASIL-RETRAINING.md) ·
[model card HuggingFace](docs/MODEL-CARD.md).

## Lisensi

- Kode proyek ini: **Apache-2.0** (mengikuti upstream Laya).
- Bobot model hasil fine-tune: **CC-BY-SA-4.0** — data training NusaX bersifat share-alike;
  lihat [docs/MODEL-CARD.md](docs/MODEL-CARD.md) untuk catatan lisensi penuh
  (termasuk kaveat non-komersial NLLB pada subset intent jv/su).
