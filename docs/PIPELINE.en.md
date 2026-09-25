# Path A — Fine-tuning Laya Multilingual for Indonesian, Javanese, Sundanese, English

**Bahasa / Language**: [Indonesia](PIPELINE.md) | **English**

This project retrains **`laya-multilingual`** (non-generative decision model, 322M, Apache-2.0)
so its *decision/classification* capability is strong on Indonesian (`id`), Javanese (`jv`),
Sundanese (`su`), and English (`en`) input. Full research report: [RISET-LAYA.en.md](RISET-LAYA.en.md).

## ✅ Results (Sep 24, 2026, model: [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1), local `runs/laya-idjvsuen-v1`)

Full tables: [HASIL-RETRAINING.en.md](HASIL-RETRAINING.en.md). Accuracy summary:

| Task | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| 60-class intent (4-language average) | 35.5% | **82.4%** | +46.8 |
| — Indonesian / English | 41.0% / 47.9% | **86.2% / 85.7%** | +45 / +38 |
| — Javanese / Sundanese (NLLB-translated) | 28.1% / 25.2% | **80.5% / 77.2%** | +52 / +52 |
| Sentiment (4-language average) | 61.4% | **82.7%** | +21.3 |
| — Sundanese (NusaX, human-written text) | 41.5% | **75.7%** | +34.2 |
| Code-switching (average) | 35.6% | **82.9%** | +47.3 |

ECE dropped on every set (e.g. intent id 0.213→0.064); per-type calibration temperatures were
fitted. Honest caveat: the jv/su intent & CS tests use translated/synthetic text (not human
ground truth); the honest human-text numbers are the NusaX (sentiment) rows. Typed-decision
ability outside these two tasks was not re-measured (possible forgetting — see TODO).

## Setup

```bash
# venv lives in .venv (Python 3.13, torch 2.11 cu128)
.venv/Scripts/python.exe -m pip install -r requirements.txt   # if rebuilding
```

## Pipeline

| Step | Script | Output | Time (RTX 5050 8GB) |
|---|---|---|---|
| 1. Download raw data | `scripts/01_download_data.py` | `data/raw/` (MASSIVE id+en, NusaX-senti 4 langs) | ~5-10 min |
| 2. (Optional) Translate id→jv/su | `scripts/02_translate_jav_sun.py` | `data/translated/massive-{jv,sun}.jsonl` (NLLB-600M) | ~20-40 min |
| 3. Build typed-decisions dataset | `scripts/03_build_dataset.py` | `data/processed/{train_items,test_sets}.pt` | ~2-5 min |
| 4. Baseline evaluation | `scripts/05_eval.py --tag baseline` | `data/processed/eval_baseline.json` | ~5-15 min |
| 5. Training | `scripts/04_train.py` | `runs/laya-idjvsuen-v1/` | ~2 h (3 epochs) |
| 6. Evaluate results | `scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned` | `eval_finetuned.json` | ~5-15 min |
| 7. Compare | `scripts/06_compare.py` | `docs/HASIL-RETRAINING{,.en}.md` | seconds |
| 8. Publish to HF | `scripts/07_publish_hf.py --repo-id <user>/...` | HuggingFace repo | minutes |

## Stage 2 (optional) — internal ticket-domain adaptation

Further adaptation on real ticket conversations from two internal databases
(PostgreSQL + MySQL; credentials in the gitignored `secrets/db.ini`):

| Step | Script | Purpose |
|---|---|---|
| 9. Fetch data | `scripts/08_fetch_tickets.py` | pull the ticket message table from both DBs into raw JSONL |
| 10. Preprocess | `scripts/09_preprocess_tickets.py` | normalize, mask PII/secrets, dedup, user/handler split, language stats |
| 11. Production eval | `scripts/10_eval_tickets.py` | test the model on real message samples |
| 12. Weak labeling | `scripts/11_weak_label_tickets.py` | auto-label 12 domain categories (id+en keyword rules) |
| 13. v2 dataset | `scripts/12_build_ticket_intent.py` | 12-category items + per-ticket split + v1 replay |
| 14. v2 training | `scripts/04_train.py --model runs/laya-idjvsuen-v1 --data data/processed/train_items_v2.pt --out runs/laya-idjvsuen-v2 --model-name ...` | ~30 min |

Domain categories (12): Asset & Devices, Infrastructure, Platforms, Employee Support,
Security, Compliance, Account & Identity, Data & Reporting, Networks & Connectivity,
Application Issue, Service, Request Access. Honest note: this stage's labels are
*weak labels* (keywords, not human annotation) — the metric is "weak-label accuracy".

### Running

```bash
cd D:\MyTools\laya
.venv/Scripts/python.exe scripts/01_download_data.py
.venv/Scripts/python.exe scripts/02_translate_jav_sun.py      # jv/su augmentation (optional but recommended)
.venv/Scripts/python.exe scripts/03_build_dataset.py
.venv/Scripts/python.exe scripts/05_eval.py --tag baseline    # pre-training numbers
.venv/Scripts/python.exe scripts/04_train.py                  # --resume to continue; --optim adamw8bit if OOM
.venv/Scripts/python.exe scripts/05_eval.py --model runs/laya-idjvsuen-v1 --tag finetuned
```

## Design

**Data** (all labeled, primary sources):

| Task | Languages | Train | Test | Source |
|---|---|---|---|---|
| Intent (60 classes) | id | 14,507 (train+dev) | ≤2,000 | MASSIVE 1.1, CC BY 4.0 |
| Intent | en | 14,507 | ≤2,000 | MASSIVE 1.1 |
| Intent | jv, su | 6,000/lang | ≤2,000/lang | NLLB translation from id (synthetic) |
| Sentiment (3 classes) | id, jv, su, en | 500/lang | 400/lang | NusaX-senti (manual, CC-BY-SA-4.0) |
| Code-switching (eval only) | id+en, id+jv, id+su | — | 500/combo | synthetic parallel half-splices |

**Training** — the official `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` notebook recipe,
adapted to 1 GPU: RLCD (policy gradient + proper scoring rules, group baseline) + supervised
soft cross-entropy, encoder LR 2.5e-5 / head 1e-4, cosine decay + warmup, bf16 autocast,
gradient checkpointing (encoder + head), effective batch 64, 3 epochs, then **per-question-type
calibration temperature fitting** on a 400-item hold-out.

**Config changes vs the original checkpoint**: `max_len` 1024→768, `head_max_len` 256→512
(the 60 intent options are no longer truncated to identical 3-token prefixes). The training
run writes these values into the checkpoint config, so evaluation via `laya.load()` uses the
same budget.

**Evaluation** — accuracy, macro-F1, ECE, mean confidence, coverage@0.8 per (task, language);
the forward path is identical to the `Agent.predict` inference route (temperatures from the
config), without the Router (the multilingual checkpoint is loaded directly — avoiding the
PR #286 routing bug).

## Inference with the fine-tuned model

```python
import laya
agent = laya.load(r"D:\MyTools\laya\runs\laya-idjvsuen-v1")
r = agent.predict("Waktune sindur enak banget", {
  "sentiment": {"type": "choice", "instructions": "What is the sentiment of the text?",
               "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}
})
```

## Important notes

- **The jv/su intent data is machine translation** (NLLB) — good for language adaptation, but
  the jv/su intent evaluation numbers are indicative, not human ground truth. The primary
  metrics remain the original id/en test sets + NusaX-senti (native-speaker manual).
- **Licensing**: model Apache-2.0; MASSIVE CC BY 4.0; NusaX **CC-BY-SA-4.0 (share-alike)** —
  a model trained on it must be shared under a compatible license when distributed.
- **Kaggle fallback** if the local GPU is insufficient: the official 2×T4 notebook — upload
  the `train_items.pt` from step 3 and change MODEL_ID to the multilingual path
  (`convaiinnovations/laya-multilingual`, encoder `jhu-clsp/mmBERT-base`).
- VRAM: full fine-tune of 322M ≈ 6-6.5 GB (fp32 master + AdamW + bf16 grad-ckpt activations).
  If OOM: `--micro-batch 4` or `--optim adamw8bit`.
