# laya-idjvsuen

**Language**: [Indonesia](README.md) | **English**

A fine-tune of [ConvAI Innovations Laya](https://huggingface.co/convaiinnovations/laya-multilingual)
(a non-generative *decision model*, Apache-2.0) so its classification/decision capability works
well on **Indonesian, Javanese, Sundanese, English, and mixed (code-switched) input**.

> Laya is a decision model (typed choice/score/noul questions with calibrated probabilities),
> not a generative model — it never produces text, and this project does not change that.
> Upstream SDK & code: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (not vendored
> in this repo; `pip install laya`).

## Results (accuracy, baseline → fine-tuned)

| Task | Baseline | Result |
|---|---|---|
| 60-class intent — id / en (MASSIVE) | 41.0% / 47.9% | **86.2% / 85.7%** |
| 60-class intent — jv / su (NLLB-translated) | 28.1% / 25.2% | **80.5% / 77.2%** |
| Sentiment — id / jv / su / en (NusaX) | 41–74% | **76–87%** |
| Code-switch id+en / id+jv / id+su (synthetic) | 34–38% | **82–85%** |

ECE (calibration error) drops on every test set. The model lives in `runs/laya-idjvsuen-v1`.

## Usage

```bash
pip install laya
```

```python
import laya
agent = laya.load("runs/laya-idjvsuen-v1")  # or "<hf-username>/laya-idjvsuen-v1"
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
python scripts/01_download_data.py     # MASSIVE id/en + NusaX-senti (4 languages)
python scripts/02_translate_jav_sun.py # jv/su augmentation via NLLB (optional)
python scripts/03_build_dataset.py     # typed-decisions items
python scripts/05_eval.py --tag baseline
python scripts/04_train.py             # RLCD + CE, ~2 h on an 8 GB GPU
python scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned
python scripts/06_compare.py           # baseline vs results table
python scripts/07_publish_hf.py --repo-id <user>/laya-idjvsuen-v1  # publish
```

Stage 2 (optional) — internal ticket-domain adaptation (scripts 08-12, data pulled from
internal databases via `secrets/db.ini`): see [docs/PIPELINE.en.md](docs/PIPELINE.en.md).

## Full documentation

Everything detailed lives in [`docs/`](docs/), available in both languages (id + en):

| Document | Indonesia | English |
|---|---|---|
| Deep research | [RISET-LAYA.md](docs/RISET-LAYA.md) | [RISET-LAYA.en.md](docs/RISET-LAYA.en.md) |
| Pipeline guide | [PIPELINE.md](docs/PIPELINE.md) | [PIPELINE.en.md](docs/PIPELINE.en.md) |
| Results & evaluation | [HASIL-RETRAINING.md](docs/HASIL-RETRAINING.md) | [HASIL-RETRAINING.en.md](docs/HASIL-RETRAINING.en.md) |
| HF model card | [MODEL-CARD.md](docs/MODEL-CARD.md) | [MODEL-CARD.en.md](docs/MODEL-CARD.en.md) |

## License

- This project's code: **Apache-2.0** (matching Laya upstream).
- Fine-tuned model weights: **CC-BY-SA-4.0** — the NusaX training data is share-alike;
  see [docs/MODEL-CARD.en.md](docs/MODEL-CARD.en.md) for full licensing notes
  (including the NLLB non-commercial caveat on the jv/su intent subset).
