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

**Language**: **English** | [Indonesia](MODEL-CARD-V4.md)

**General fine-tune of [laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1)**
(itself a multilingual fine-tune of
[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)):
a non-autoregressive decision model (mmBERT-base, 322M parameters) producing calibrated
decisions on **Indonesian (id), Javanese (jv), Sundanese (su), English (en), and mixed
(code-switching) input** — extended in a **general** direction (not the ticket domain):
**7-category topic classification** plus IndoNLU tasks (emotion, review sentiment, aspect),
with **full replay of the v1 data** so every existing skill holds or improves.

| Model | Focus | Link |
|---|---|---|
| laya-idjvsuen-v1 | general multilingual (MASSIVE 60-class intent + NusaX sentiment) | [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) |
| laya-idjvsuen-v3 | v1 + 12 ticket categories (for ticket routing) | [faall7479/laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3) |
| **laya-idjvsuen-v4** | **generalization from v1**: + SIB-200 topics + IndoNLU, full replay — **for general classification** | this repo |

**This model does not generate text** — it answers caller-defined typed questions
(choice/score/noul) with calibrated probabilities in a single forward pass. Built on the
work of [ConvAI Innovations](https://huggingface.co/convaiinnovations) with the open-source
SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).

**Author**: muhfalihr ([github.com/muhfalihr](https://github.com/muhfalihr))

## Usage

```bash
pip install laya
```

```python
import laya, json, urllib.request

agent = laya.load("faall7479/laya-idjvsuen-v4")

# NEW task: topic classification (definitions exactly as trained)
qd = json.load(urllib.request.urlopen(
    "https://huggingface.co/faall7479/laya-idjvsuen-v4/raw/main/question_defs_general.json"))
q = {"t": {"type": "choice",
           "instructions": qd["topic"]["instructions"],
           "criteria": qd["topic"]["criteria"]}}

r = agent.predict("Timnas berjuang di laga pamungkas kualifikasi", q)
# r["answers"]["t"]["choice"] -> "sports" | probabilities | answer_confidence
```

The v1 skills (MASSIVE 60-class intent, NusaX 3-class sentiment — `question_defs.json` in
this repo) remain available and did not regress.

## Training data

| Task | Languages | Train | Source | License |
|---|---|---|---|---|
| MASSIVE intent (60 classes) — replay | id, en, jv, su | 13,547+13,547 id/en; 7,000 jv/su (NLLB MT) | MASSIVE 1.1 + NLLB | CC-BY-4.0 / see v1 card |
| NusaX sentiment (3 classes) — replay | id, jv, su, en | 500/language | manual annotation by native speakers | CC-BY-SA-4.0 |
| **SIB-200 topics (7 classes)** | id, jv, su, en | 800/language (parallel FLORES-200) | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC-BY |
| **IndoNLU: emotion/review-sentiment/aspect** | id | ~12k items | [indonlp/indonlu](https://huggingface.co/datasets/indonlp/indonlu) (emot, smsa, casa) | varies per sub-dataset |

Synthetic code-switching is used for evaluation only. Trained from the
laya-idjvsuen-v1 weights on a **free Google Colab T4** (fp16 + GradScaler, adamw8bit,
effective batch 64, 2 epochs ≈ 3 hours); fitted temperature: choice 3.55.

## Evaluation results (v1 → v4, identical test sets)

**New capability — 7-category topics (SIB-200):**

| Set | v1 | v4 | ECE v1→v4 |
|---|---|---|---|
| Topics — id / en | 78.4% / 77.0% | **87.3% / 89.2%** | 0.157→0.080 / 0.165→0.066 |
| Topics — jv / su | 72.1% / 66.2% | **84.3% / 79.9%** | 0.159→0.104 / 0.126→0.111 |
| Topic code-switching (6 combos) | 71.3–78.4% | **88.2–92.0%** | 0.149–0.185 → 0.046–0.066 |

Coverage@conf≥0.8 on topics rises from ~45% (v1) to ~90% (v4) — automation-grade.

**Existing skills — full replay, no regression (all improved):**

| Set | v1 | v4 |
|---|---|---|
| Intent id / en | 86.2% / 85.7% | **87.3% / 87.6%** |
| Intent jv / su | 80.5% / 77.2% | **82.3% / 80.0%** |
| Sentiment id / jv / su / en | 86.0% / 81.8% / 75.7% / 87.3% | **94.0% / 86.0% / 82.0% / 89.7%** |
| Intent code-switching (3 combos) | 81.6–85.2% | **83.4–86.4%** |

The +8-point Indonesian-sentiment jump is consistent with transfer from `smsa`
(Indonesian reviews) in IndoNLU. Methodology & the full 21-test-set table:
`eval_v4-general.json` in the [pipeline repo](https://github.com/muhfalihr/laya-idjvsuen).

## Limitations

1. SIB-200 jv/su sentences come from FLORES-200 translations — indicative, not native-speaker
   annotation (same caveat as NLLB on the v1 card); the most honest human-text rows remain NusaX.
2. The IndoNLU eval sets were not re-run locally (legacy loader; training itself ran on
   Colab) — their effect shows indirectly through the Indonesian-sentiment gain.
3. Not trained on ticket data — for internal ticket routing use
   [laya-idjvsuen-v3](https://huggingface.co/faall7479/laya-idjvsuen-v3).
4. Not a generative model; calibration fitted on this task distribution.

## License & attribution

- Fine-tuned weights: **CC-BY-SA-4.0** (NusaX share-alike lineage; NLLB NC caveat on the
  jv/su subset as documented on the v1 card).
- Base: [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) →
  [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0) © ConvAI Innovations; SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).
- MASSIVE 1.1 © Amazon CC-BY-4.0; NusaX-senti © IndoNLP CC-BY-SA-4.0;
  SIB-200 (Adelani et al., EACL 2024) CC-BY; IndoNLU (Wilie et al., AACL 2020) per-sub-dataset licenses.

## Citation

```bibtex
@misc{laya-idjvsuen-v4,
  title  = {laya-idjvsuen-v4: general fine-tune of the multilingual Laya decision model — SIB-200 topics and IndoNLU tasks for Indonesian, Javanese, Sundanese, English, and code-switching},
  author = {muhfalihr},
  year   = {2026},
  note   = {Fine-tune of faall7479/laya-idjvsuen-v1 with full v1 replay (no forgetting) plus SIB-200 topics and IndoNLU; trained on a free Colab T4},
  url    = {https://huggingface.co/faall7479/laya-idjvsuen-v4}
}
```
