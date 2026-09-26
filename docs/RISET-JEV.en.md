# Deep Research: Jev (TypeSafe AI) & Gap Analysis vs laya-idjvsuen

**Language**: English | [Indonesia](RISET-JEV.md)

> Compiled 26 September 2026. All claims from primary sources (docs.typesafe.ai, the TypeSafe
> blog, evals.typesafe.ai, OpenRouter, HuggingFace API, legal/mca) unless explicitly marked.
> Supplemented with analysis of 8 attached explainer-thread screenshots by @techwith.ram.

---

## Executive Summary (read this first)

1. **Jev is the same model class as Laya** — a non-generative *System One / decision model*:
   input `state` (unstructured text) + typed questions (`choice`/`score`/`noul`), output a
   typed decision + calibrated probabilities, no text generation. Its I/O format is nearly
   identical to laya's (`build_sequence`). The difference is not model *kind* but
   **generality, scale, and ecosystem**.
2. **Jev is fully proprietary** — parameter count never disclosed, no weights, no official
   HuggingFace org, no customer fine-tuning. API only (`POST /v1/systemone`; on OpenRouter
   via `~typesafe/jev-latest`).
3. **Training method: RLCD** ("Reinforcement Learning for Calibrated Decisions") — a third
   path after RLHF/RLVR. Corpus, data size, and loss are unpublished. The laya-idjvsuen
   pipeline (RLCD REINFORCE + proper scoring rules + soft CE) is conceptually the same recipe.
4. **Jev is not the smartest on narrow domain tasks**: TypeSafe's own evals put Jev at 67.8%
   overall — **behind frontier models** (sol 74.1%) and only 76.0% on Customer Service (vs
   sol 78.3%). Jev's edge is **price/latency** (orders of magnitude) and cross-domain
   generality. This is consistent with our benchmark: laya-idjvsuen-v3 93.3% vs Jev 84.7% on
   the 12-category internal ticket-routing set.
5. **Jev admits it is weaker in non-English languages** — the natural moat of laya-idjvsuen
   (id/jv/su/code-switching).
6. **⚠️ Legal**: TypeSafe's ToS (MCA §2.3(b)) **prohibits distilling from Jev outputs**. Never
   train any model on Jev outputs. Benchmarking and publishing evaluation results is allowed
   (no clause forbids it).
7. **Short answer to "can laya-idjvsuen become as smart as Jev?"**: *on the domains it is
   trained on — this already happened and can be widened; as a general cross-domain decision
   model — not with a 322M backbone, that needs a 1–9B+ base; copying Jev from its outputs —
   forbidden by ToS.* Full roadmap: [Section 8](#section-8--can-laya-idjvsuen-become-as-smart-as-jev).

---

## Section 1 — What Jev is

### 1.1 Identity

| Aspect | Fact (primary source) |
|---|---|
| Developer | TypeSafe AI — CEO **Diogo Almeida** (co-inventor of RLHF/InstructGPT, ex-Google Brain), CTO Erik Gafni, COO Sasha Sheng. Out of stealth 15 Sep 2026 ($40M seed — secondary source, not yet verified primary) |
| Version | Only official release: **`jev-1.13.0`** (alias `jev-latest`; `jev-preview` also points there) |
| Parameters | **Not disclosed** (blog says "a new model architecture, parallel sampler" with no detail) |
| Access | Hosted API only: `POST /v1/systemone`; OpenRouter `~typesafe/jev-latest`, `typesafe/jev-router` |
| Weights | **None** — HF orgs `typesafe`/`typesafeai`/`typesafe-ai` are all empty |
| Customer fine-tuning | **Not available** |

### 1.2 Technical specs vs laya

| Capability | Jev 1.13 | laya / laya-idjvsuen |
|---|---|---|
| Primitives | Choice (max **255** options), Score (rubric levels), Noul | Choice, Score, Noul — same |
| Questions per call | Many, parallel ("adding questions barely changes response time") | Many (per-qid loop; one forward per sequence) |
| Context | **64k tokens** (32k state + questions) | 768–1024 tokens (our run config: 768) |
| Structured criteria | Yes — options/levels/instructions accept JSON | Yes — `criteria` dict/list |
| Modality | Text only | Text only |
| Output | `choice`/`score` + `probabilities` + `confidence` | same + per-type temperature calibration |
| Claimed latency | 70–500 ms | local, hardware-dependent (~hundreds of ms on consumer GPU) |
| Self-host | No | **Yes** — Apache-2.0 / CC-BY-SA-4.0 |

### 1.3 "Open-Jev" 2B/9B/27B is not a TypeSafe product

An independent reproduction project by Zefan Cai (Qwen3.5/3.8 base + decision head, rank-8
LoRA for 27B). Code MIT, adapters Apache-2.0, public datasets `Open-Jev` / `Open-Jev-v1.1`
(**326,619–408,884 rows** of typed decisions from 25 task sources) — legal to train on
(see Section 8). Third-party JevBench: Jev 1.13 = 86.6% overall / 73.0% Hard;
Open-Jev-27B-v1.1 = 85.3% / 72.1% Hard.

---

## Section 2 — Capability & benchmarks

### 2.1 TypeSafe's official evals (evals.typesafe.ai; reference labels = mean of GPT-6 Astra & Claude Fable 5.1)

| Model | Overall | Security | Agent Trace | Invoice | Customer Service |
|---|---|---|---|---|---|
| **Jev** | **67.8%** · $0.0004 · 0.4s | 61.7% | 71.6% | 61.8% | 76.0% · $0.0001 |
| sol (GPT) | **74.1%** | 62.5% | **76.6%** | **79.1%** | **78.3%** |
| opus 5 | 73.1% | **66.2%** | 75.2% | 78.4% | 72.4% |
| terra (GPT) | 67.9% | 51.2% | 73.0% | 74.7% | 72.7% |
| sonnet 5 | 67.8% | 60.8% | 68.0% | 72.9% | 69.3% |
| DS v4 flash | 64.4% | 37.9% | 73.0% | — | 76.8% |
| haiku 4.5 | 53.6% | 58.8% | 57.2% | 42.9% | 55.4% |

Key pattern: **Jev wins on cost/latency (claimed 193.6× faster, 444.6× cheaper), loses on
average accuracy to frontier models.** Homepage: $0.000081/workflow (0.114s) vs $0.013880
(8.566s) for an LLM. Official pricing: **$42 per billion input tokens, output free**; rate
limits 250k tokens/s, 1,200 req/min.

### 2.2 Jaggedness — 9 failure modes TypeSafe itself documents (docs.model-jaggedness)

1. **Literal reading** — answers the question as written, not as meant (negation/scoping read literally).
2. **Not a calculator** — unreliable counting, weak hex/RGB, poor numeric calibration.
3. **Dates as text** — not read as ordered quantities; quarters/windows are weak.
4. **Indirection** — double-negative/complex instructions less reliable.
5. **Context rot** — accuracy drops when the state contains irrelevant content.
6. **Not adversarial by default** — injected instructions "can move the answer".
7. **Contradictory instructions/criteria** degrade performance.
8. **Structural invariants not guaranteed** — real example: Noul refund 0.22 vs Choice no 0.99
   (confidence 0.97); refund/not_refund pair 0.72/0.47 (sums to 1.19 — violates P/1−P).
9. **Chained choices for text generation** "will not work well and will be very slow".

Plus models.md: **non-English languages are weaker** (CJK explicitly called out).

### 2.3 Claim credibility

TypeSafe is open about its own biases: evals run from laptops on the West Coast, reference
labels are GPT/Claude products, workflow gains "on the higher end", and the 0% hallucination
claim "is not empirical" (guaranteed by schema, not measured). No comparison vs Laya exists
in any primary source. One public-circulation correction: TypeSafe's founder is Diogo
Almeida, **not** Joseph Perla (he founded TrustedRouter, a reseller of access to Jev).

### 2.4 From the attached screenshots (@techwith.ram explainer thread)

- **Slide 07**: "cannot hallucinate" needs context — schema errors (invalid label) are
  impossible, but **decision errors** (valid but wrong label) still happen. *Valid ≠ correct.*
- **Slide 08**: Jev does not replace the LLM — **it wraps it** (choose model → LLM works →
  approve tool call → verify result). Jev handles decisions, the LLM handles reasoning.
- **Slide 09**: Jev's sweet spot = **semantic + bounded + repeated** decisions (routing,
  triage, rerank, guardrails, high-volume labeling). Open-ended → LLM; deterministic rules → code.
- **Slide 06**: the headline value is **probabilities + confidence**, not just answers —
  thresholds enable automation (the same selling point as laya).

---

## Section 3 — Data & training

- Official method: **RLCD** — positioned as a third path after RLHF (sycophancy, confident
  hallucination) and RLVR (slow/expensive). Output contract: the model does not generate
  text, only decisions + probabilities; group-level calibration (0.2 → happens ~20% of the time).
- **Unpublished**: corpus, data size, detailed loss, SFT pipeline, distillation. The blog has
  an FAQ heading "What is Jev trained on?" — the answer is empty. No paper/arXiv.
- No training on customer data; enterprise ZDR available.

---

## Section 4 — Ecosystem

Python SDK (v0.5.7 → v0.7.1 within one week) + JavaScript SDK (~30 pages), agent skills for
Claude Code/Codex, a GitHub adapter, the console.typesafe.ai playground, a public evals
site, ~20 cookbooks (CLERC re-ranking 5%→18% top-1, guardrails, hierarchical classification
of 75 industries).

---

## Section 5 — Compliance & legal (IMPORTANT)

- **Benchmarking is allowed** — no ToS clause forbids evaluating or publishing results.
- **§2.3(b) of the Master Customer Agreement PROHIBITS**: using the Services/Outputs to
  *"perform model distillation, train a model to imitate the output, or develop a similar or
  competing product or service"*. Practical consequences:
  1. **Never train laya-idjvsuen (or any model) on Jev outputs.**
  2. Our benchmark script (13) is fine — that is evaluation, not training.
  3. The "competing product" clause is potentially broad if you become a TypeSafe customer —
     get legal advice before commercializing.

---

## Section 6 — Where laya-idjvsuen stands today (our data)

| | laya-multilingual (base) | **laya-idjvsuen-v3** | Jev 1.13 |
|---|---|---|---|
| 12-category internal tickets (identical 300 samples) | 24.7% | **93.3%** (ECE 0.015) | 84.7% (ECE 0.083) |
| Cost per 1k decisions | free (local) | free (local, 8GB GPU) | ~$0.0093 in our benchmark ($0.042/MTok input) |
| Avg latency (our benchmark) | local | local | 0.32s |
| id/jv/su/CS languages | weak | **strong** | weak (self-admitted) |
| Cross-domain generality | — | medium (MASSIVE 60-class + domain) | **strong** |
| Self-host / internal-data privacy | yes | **yes** | no |

v3 beating Jev on a narrow domain is **consistent** with TypeSafe's own evals (Jev trails
frontier models on Invoice/Customer Service) — not an anomaly.

---

## Section 7 — What makes Jev "smart"

Three sources of Jev's edge, and which are replicable:

1. **Generality (partially replicable)** — trained on a large cross-domain decision corpus.
   Open equivalent: the `Open-Jev` dataset (400k+ rows, 25 task sources, permissive
   licensing) + MASSIVE + NusaX + your own domain data.
2. **Backbone scale (not replicable at 322M)** — Jev's parameter count is undisclosed, but
   its cross-domain reasoning implies a much larger base and/or massive pretraining.
   Open-Jev needed 9–27B to approach Jev on JevBench Hard (2B: 46/111, 9B: 66/111, 27B:
   80/111 vs Jev 81/111).
3. **Serving engineering (replicable)** — the 70–500ms latency and extreme price point are
   inference optimizations, not intelligence.

---

## Section 8 — Can laya-idjvsuen become "as smart as Jev"?

An honest answer, in three tiers:

### (1) Yes, via data/training alone — partly already achieved

- **The id/jv/su/en + CS domain**: already beats Jev (93.3% vs 84.7%). The recipe — domain
  supervised data + RLCD proper-scoring-rules — is essentially what TypeSafe claims to do.
- **Widen the domain safely & legally**: mix the public `Open-Jev` dataset (326–408k typed
  decisions, MIT/Apache) into a v4 pipeline together with replay of existing data →
  generality rises without touching Jev outputs.
- **Beat Jev's self-admitted weaknesses, in-domain**: negation augmentation (vs literal
  reading), mixed-format date/number data, contradictory criteria, **prompt-injection
  robustness** (train with injected instructions — Jev fails here), long states +
  distractors (vs context rot).
- **Calibration**: our proper scoring rules are more explicit than anything TypeSafe
  documents; gains come from data coverage + hard negatives, not architecture.
- **A human audit set** for the 12 categories (long-standing TODO) → claims get harder than
  "learned the annotator".

### (2) Requires scale/architecture change — possible, but not "another fine-tune"

- **Jev-level generality**: a 322M backbone will not catch up on world knowledge. Realistic
  path: move to a 1–9B base (the Open-Jev pattern: LoRA + decision head on top of Qwen) —
  but our 8GB GPU only fits 1–2B (LoRA, quantized), and that is a different class of
  project, not a "v4 update".
- **64k context**: needs a long-context encoder or chunking+aggregation (can be simulated at
  the pipeline level: summarize state first → decide).
- **Parallel multi-question per forward** and cardinality 255: head modifications —
  conceptually simple (our 512 head_max_len already hosts 60 options) but not free.
- **Consistent sub-second latency**: INT8/ONNX quantization + batching at serving time.

### (3) Unrealistic / not allowed

- **"As smart as Jev in general" at 322M** — no amount of fine-tuning gets there; that is a
  capacity limit, not a data problem.
- **Copying Jev from its outputs** — **forbidden by TypeSafe ToS §2.3(b)**. Don't.
- **Matching its ecosystem** (two SDKs, public evals, cookbooks, integrations) — years of
  product work, not model capability.

### Prioritized recommendations (concrete)

1. **v4 = generality**: mix the public `Open-Jev` dataset (legal) + MASSIVE/NusaX replay +
   ticket data → raise cross-domain capability without losing the domain.
2. **Hardening**: negation + prompt-injection + dates/numbers augmentation → directly
   attacks 4 of Jev's 9 documented jaggedness failures.
3. **Human audit** of 200–300 tickets → replace weak labels in the test set → benchmark
   numbers become defensible for publication.
4. **Serving**: ONNX/quantized export for latency and CPU inference.
5. (Optional, long-term) **a 1–2B backbone** with LoRA + decision head on the 8GB GPU if you
   truly want to chase generality — with the honest expectation that this is a new project
   class.

---

## Primary sources

- Blog: <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- Model docs: <https://docs.typesafe.ai/models.md> · primitives: <https://docs.typesafe.ai/primitives.md>
- Jaggedness: <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>
- Evals: <https://evals.typesafe.ai/> · Legal/MCA: <https://typesafe.ai/legal/mca>
- OpenRouter: <https://openrouter.ai/~typesafe/jev-latest>
- Open-Jev (community, not TypeSafe): <https://zefan-cai.github.io/open-jev> ·
  [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) ·
  [github.com/Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)
- Team: <https://typesafe.ai/team>

*Not found in primary sources: Jev's parameter count, detailed architecture, training
corpus/size, the blog FAQ answers (headings exist, content empty), a paper, any official
comparison vs Laya, per-token pricing on OpenRouter (endpoints array empty when checked).*
