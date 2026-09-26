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

**Bahasa / Language**: [Indonesia](MODEL-CARD.md) | **English**

**Domain fine-tune of [laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1)**
(which is itself a multilingual fine-tune of
[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)):
a non-autoregressive decision model (mmBERT-base, 322M parameters) for calibrated typed
decisions on **Indonesian (id), Javanese (jv), Sundanese (su), English (en), and code-switched**
input — extended with an **internal ticket-domain 12-category intent** task.

| Model | Focus | Link |
|---|---|---|
| laya-idjvsuen-v1 | general multilingual (MASSIVE intent 60-class + NusaX sentiment) | [faall7479/laya-idjvsuen-v1](https://huggingface.co/faall7479/laya-idjvsuen-v1) |
| **laya-idjvsuen-v3** | v1 **+ ticket-domain 12 categories** (recommended for ticket routing) | this repo |
| laya-idjvsuen-v4 | generalization from v1: + SIB-200 topics + IndoNLU (recommended for general classification) | [faall7479/laya-idjvsuen-v4](https://huggingface.co/faall7479/laya-idjvsuen-v4) |

**The model never generates text** — it answers caller-defined typed questions
(choice/score/noul) with calibrated probabilities in a single forward pass. It builds on the
work of [ConvAI Innovations](https://huggingface.co/convaiinnovations) with the open-source
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) SDK (Apache-2.0).

**Author**: muhfalihr ([github.com/muhfalihr](https://github.com/muhfalihr))

## The 12 ticket-domain categories

Asset & Devices · Infrastructure · Platforms · Employee Support · Security · Compliance ·
Account & Identity · Data & Reporting · Networks & Connectivity · Application Issue ·
Service · Request Access

## Usage

```bash
pip install laya
```

```python
import laya, json, urllib.request

agent = laya.load("faall7479/laya-idjvsuen-v3")

# exact question definition used in training (12 categories, correct option order)
qd = json.load(urllib.request.urlopen(
    "https://huggingface.co/faall7479/laya-idjvsuen-v3/raw/main/question_defs.json"))
q = {"k": {"type": "choice",
           "instructions": qd["ticket_intent"]["instructions"],
           "criteria": qd["ticket_intent"]["criteria"]}}

r = agent.predict("mas tolong buatkan bucket s3 baru untuk data riset", q)
# r["answers"]["k"]["choice"] -> "infrastructure" | probabilities | answer_confidence
```

The model retains v1's general tasks (60-class MASSIVE intent, 3-class NusaX sentiment) —
see the v1 card for those question definitions.

## Training data

| Task | Languages | Train | Source | License |
|---|---|---|---|---|
| MASSIVE intent (60 classes) | id, en, jv, su | 13,547+13,547 id/en; 7,000 jv/su (NLLB MT) | MASSIVE 1.1 + NLLB | CC-BY-4.0 / see v1 card |
| NusaX sentiment (3 classes) | id, jv, su, en | 500/lang | native-speaker annotation | CC-BY-SA-4.0 |
| **Ticket intent (12 classes)** | id (colloquial + code-mixed) | 5,731 (incl. rare-class oversampling ×3) | real internal support tickets, **keyword weak labels** + relaxed rare-class mining | internal |

Total 17,731 items; v1 replay (12,000 items) prevents forgetting; 400-item hold-out for
temperature calibration. Trained from laya-idjvsuen-v1 weights with the official Laya RLCD
recipe (2 epochs, single 8 GB GPU, ~30 min); fitted temperature: choice 3.44.

## Evaluation results

12-category ticket test (1,126 real messages, strict weak labels):

- Overall accuracy **94.0%**, macro-F1 0.782, ECE 0.019.
- Per-class gains over v2 (pre-boost): platforms 0.36→0.64, employee_support 0.75→1.00,
  data_reporting 0.80→1.00, networks 0.58→0.79; macro per-class average 0.830→0.886.

### Benchmark vs Jev (OpenRouter) — identical 300 real samples

![Overall benchmark](assets/bench_overall.png)

![Per-category matrix](assets/bench_perclass.png)

v3 reaches **93.3% accuracy / ECE 0.015** vs Jev 1.13 at 84.7% — running locally with zero
API cost. Methodology and honest caveats: `BENCHMARK.en.md` in the
[pipeline repo](https://github.com/muhfalihr/laya-idjvsuen).

Multilingual regression (15 test sets vs v1): intent id/en/jv/su within 1.6 pts, Sundanese
sentiment +3.8 pts, code-switching stable — replay prevented forgetting.

## Limitations

1. **Ticket labels are weak labels** (keyword rules, not human annotation) — part of the
   accuracy is agreement with the rule annotator. A human-audited sample is recommended
   before production SLAs.
2. **Security & compliance remain data-starved** (21 and 111 training samples): phishing is
   caught, but subtler incidents may fall to `account_identity`. Collecting ~50-100 real
   historical cases per class is the highest-value next step.
3. jv/su MASSIVE intent data is machine-translated (NLLB, CC-BY-NC caveat — see v1 card);
   honest human-text numbers are the NusaX rows.
4. Not a generative model; calibration fitted on this task distribution.

## License & attribution

- Fine-tuned weights: **CC-BY-SA-4.0** (NusaX share-alike lineage; NLLB NC caveat on the
  jv/su subset as documented in the v1 card).
- Base: [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0) © ConvAI Innovations; SDK [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (Apache-2.0).
- MASSIVE 1.1 © Amazon CC-BY-4.0; NusaX-senti © IndoNLP CC-BY-SA-4.0.

## Citation

```bibtex
@misc{laya-idjvsuen-v3,
  title  = {laya-idjvsuen-v3: domain fine-tune of the multilingual Laya decision model for Indonesian, Javanese, Sundanese, English, code-switching, and internal ticket routing},
  author = {muhfalihr},
  year   = {2026},
  note   = {Fine-tune of faall7479/laya-idjvsuen-v1 (Apache-2.0 lineage) on MASSIVE 1.1, NusaX-senti, NLLB translations, and real internal ticket data},
  url    = {https://huggingface.co/faall7479/laya-idjvsuen-v3}
}
```
