# Deep Research: The Laya Model & Retraining Paths for Indonesian, Regional Languages, English, and Code-Switching

**Bahasa / Language**: [Indonesia](RISET-LAYA.md) | **English**

> Compiled September 24, 2026. All claims come from primary sources (HuggingFace, GitHub, arXiv,
> ACL Anthology) except where explicitly marked "unverified". Research done by 2 parallel research
> agents, consolidated into this document.

---

## Executive Summary (read this first)

1. **⚠️ Laya is NOT a generative/conversational model.** Laya is a non-autoregressive *decision
   model* built on a BERT encoder for structured classification: it takes a state
   (text/ticket/JSON) + a typed question (`choice`, `score`, `noul`) and returns a typed answer
   plus a calibrated probability in a single forward pass. Its own model card states: *"never
   generates text, so there is nothing to parse and nothing to hallucinate"*
   ([official README](https://huggingface.co/convaiinnovations/laya/raw/main/README.md)). **There
   are no vision/audio components at all.**
2. **"Retraining Laya to support Indonesian" can only mean** improving its
   *classification/decision* capability in Indonesian — not turning it into a chatbot. If the goal
   is a multilingual conversational model (id + regional + en + mixes), the right base is a
   different generative LLM (e.g. Qwen/Gemma, or directly Sahabat-AI / SEA-LION which already
   support id+jv+su+en) — the full playbook is in [Section 5](#section-5--path-b-alternative-generative-model-language-adaptation-playbook).
3. **Current Indonesian performance is not production-ready:** the multilingual checkpoint's
   accuracy on MASSIVE (51 languages, 20 options) is only **0.51** (macro-F1 0.4955, ECE 0.305);
   its language router misdetects plain Latin-script Indonesian as English
   ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)). This means the improvement room
   from Indonesian fine-tuning is very large.
4. **Apache-2.0 license** — retraining, distributing derivative weights, and commercial use are
   fully allowed. The official fine-tune pipeline is open source and proven to run on
   **2× Kaggle T4s (free)**.
5. **Possible name confusion**: there is another "LAYA" by Articul8 AI (a 32B Sanskrit-heritage
   model, different company — [Hindustan Times](https://www.hindustantimes.com/india-news/with-laya-sol-articul8-puts-language-scholars-at-heart-of-heritage-ai-101790155154789.html)). Don't mix them up.

---

## Section 1 — What Laya actually is

### 1.1 Definition & checkpoint family

| Checkpoint | Backbone | Parameters | Context | Languages |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large + decision head | 421M | 512 (tokenizer config 8192) | English |
| `laya-multilingual` | "mmBERT-base" (bidirectional, 22 layers, hidden 768, 256k vocab) | 322M | 1024 (up to 8,192) | 51+ languages incl. `id`, `ms`, `jv` |
| `laya-typed-decisions` | fine-tune of `laya` | ~421M | — | *"English only. Use laya-multilingual for other languages."* |

Sources: [laya model card](https://huggingface.co/convaiinnovations/laya),
[laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual),
[laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions).

- Base model confirmed from config files: root uses `answerdotai/ModernBERT-large`
  ([rl_agent_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/rl_agent_config.json));
  encoder architecture `ModernBertForMaskedLM`, vocab 50,368, 28 layers, hidden 1024
  ([encoder/config.json](https://huggingface.co/convaiinnovations/laya/raw/main/encoder/config.json)).
  Multilingual: `ModernBertForMaskedLM`/`modernbert`, vocab 256,000, 22 layers, hidden 768
  ([multilingual config](https://huggingface.co/convaiinnovations/laya-multilingual/raw/main/encoder/config.json)).
  (The multilingual encoder's provenance was later found in its config: `jhu-clsp/mmBERT-base`.)
- `pipeline_tag: text-classification`; text-only — no `vision_config`/`audio_config` in any
  config ([HF API](https://huggingface.co/api/models/convaiinnovations/laya)).
- The model is very new: created Sep 18, 2026, last changed Sep 24, 2026
  ([HF API](https://huggingface.co/api/models/convaiinnovations/laya)).

### 1.2 Tokenizer — implications for Indonesian

- **Root (English)**: `PreTrainedTokenizerFast` in ModernBERT WordPiece style, vocab 50,368,
  `[CLS]/[MASK]/[SEP]/[PAD]/[UNK]` — essentially *English-centric*
  ([tokenizer_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/tokenizer/tokenizer_config.json)).
- **Multilingual**: `PreTrainedTokenizerFast`, vocab **256,000**, with Gemma-style special tokens
  (`<bos>/<eos>`, `start_of_turn/end_of_turn`)
  ([multilingual tokenizer_config](https://huggingface.co/convaiinnovations/laya-multilingual/raw/main/tokenizer/tokenizer_config.json)).
- `id`, `ms`, `jv` are officially listed in the multilingual HF language tags
  ([HF API laya-multilingual](https://huggingface.co/api/models/convaiinnovations/laya-multilingual)).
- **Token-per-word efficiency (token fertility) for Indonesian/regional languages was never
  measured or documented publicly.** The 256k Gemma-pattern vocab most likely covers Latin-script
  Indonesian text reasonably well, but this is a hypothesis — **measure it yourself**
  (fertility methodology from the [SEA-LION paper, arXiv:2504.05747](https://arxiv.org/abs/2504.05747)).
- For non-Latin scripts (e.g. Javanese script): no data at all.

### 1.3 Training method & original data

- Method: **RLCD** — *"reinforcement learning against strictly proper scoring rules"*, REINFORCE
  with a group-mean baseline (GRPO-style), log + spherical + ranked probability score rewards,
  TD(λ=1.0) for multi-turn ([README](https://huggingface.co/convaiinnovations/laya/raw/main/README.md)).
- Root training metadata: 7,313 updates, 1 epoch, 1.96 hours, `world_size: 1`,
  `fine_tuned_from_checkpoint: true` ([rl_agent_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/rl_agent_config.json)).
- **The training corpus is NOT disclosed.** The model card only references benchmark splits:
  typed-decisions (1,200 cases / 6,000 decisions), MASSIVE, XNLI, AG News, SST-5, Banking77,
  CLINC150, DAIR Emotion. Issue [#320](https://github.com/NandhaKishorM/laya/issues/320) still
  asks the maintainers to share the typed-decisions training data (unanswered).

### 1.4 Code & infrastructure

- **GitHub repo**: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — Apache-2.0,
  21,128 stars, 1,800 forks, very active (last push Sep 24, 2026)
  ([GitHub API](https://api.github.com/repos/NandhaKishorM/laya)). Contents: the `laya` Python SDK,
  `laya-ts/`, `research/` (benchmark data), `notebooks/`, `benchmarks/`, `scripts/`, `examples/`,
  `docker/`, `docs/`.
- **Official fine-tune notebook**:
  [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
  — RLCD + supervised cross-entropy, DDP `torchrun` on **2× Kaggle T4s (16 GB/GPU)**, fp16 +
  gradient checkpointing, `LR_ENCODER 2.5e-5` / `LR_HEAD 1e-4`, 4 epochs, training the encoder
  and head together, plus **post-training temperature calibration**.
- **From-scratch pretraining: no public path exists.** Only fine-tuning from a checkpoint is
  open-sourced.
- Format: safetensors; root repo totals 2.37 GB (3 checkpoints in subfolders), multilingual
  678 MB ([tree](https://huggingface.co/convaiinnovations/laya/tree/main)). **No official GGUF**
  (26 community quantizations mentioned in the card). vLLM/llama.cpp are not supported (not a
  causal LM). Package: `pip install laya` with extras `[serve]` (FastAPI), `[mcp]`,
  `[langchain]`, `[onnx]`, `[fast]`.
- No Laya-specific paper exists. The official site cites 2 founding arXiv papers
  ([2503.23303](https://arxiv.org/abs/2503.23303), [2510.01237](https://arxiv.org/abs/2510.01237))
  that do not mention Laya/ConvAI. Product page:
  [laya.convaiinnovations.com](https://laya.convaiinnovations.com).

### 1.5 Empirical evidence of the Indonesian situation (primary data)

MASSIVE 51-language × 20-option sweep ([repo research results, `research/results/cpu_51_language_sweep.json`](https://raw.githubusercontent.com/NandhaKishorM/laya/research/research/results/cpu_51_language_sweep.json)):

| Checkpoint | id accuracy | ECE | Notes |
|---|---|---|---|
| `laya` (English root) | **0.36** | 0.613 | mean confidence 0.962 — heavily overconfident |
| `laya-multilingual` | **0.51** | 0.305 | macro-F1 0.4955; accuracy@50% coverage 0.72 |

Comparators on multilingual: en 0.68; ja 0.64; weakest Amharic 0.11.

- **Router bug for Latin-script Indonesian** ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)):
  *"Indonesian reaches 0.318 instead of 0.409, because nothing in plain-ASCII Indonesian tells it
  apart from English"* — the word `di` is detected as Italian; id/ms/jv rely only on "stray list
  hits". **Required workaround: `Router(default="multilingual")` or `lang_guess="id"`.**
  Latin-script code-switching (id-en-jv) sits exactly at the weak point of a script-based router;
  non-Latin mixed-script is only handled via script presence
  ([PR #122](https://github.com/NandhaKishorM/laya/pull/122)).
- **Evidence the gap is in the encoder, not the head**
  ([issue #320](https://github.com/NandhaKishorM/laya/issues/320)): for Spanish, Laya 60.8% vs
  Qwen-4B 84.3%; when the prompt was translated to English it became 76.5% vs 74.5% — *"laya's
  gap on these prompts is mostly the encoder's non-English comprehension, not the typed-decision
  heads"*. Meaning: fine-tuning the encoder on target-language data is the right intervention point.
- Other limitations from the model card: base near-chance zero-shot typed-decisions (0.362);
  overconfidence (ECE 0.466 → 0.081 after temperature refit); `noul` can anchor to label
  ([#156](https://github.com/NandhaKishorM/laya/issues/156)); `act_probability` without signal
  ([#185](https://github.com/NandhaKishorM/laya/issues/185)); position bias on multilingual
  `score` ([#131](https://github.com/NandhaKishorM/laya/issues/131)).

### 1.6 License & community

- **Apache-2.0** — confirmed in the [HF API](https://huggingface.co/api/models/convaiinnovations/laya)
  (both repos), GitHub, and the product site (*"100% open-source Apache 2.0 weights"*). No
  non-commercial or gated clauses.
- HF: `laya` 3,168 likes; GitHub: 21,128 stars. Card claims: 42 finetunes, 26 quantizations,
  32 derivative Spaces.
- Relevant language issues: [#320](https://github.com/NandhaKishorM/laya/issues/320) (asks for
  training data & a multilingual typed-decisions checkpoint), [#286](https://github.com/NandhaKishorM/laya/pull/286)
  (id/ms/jv routing), #42, #202, #207 (Portuguese), #127 (Mandarin calibration), #113, #122
  (mixed-script).

### 1.7 Training memory estimate (standard math from parameter count)

- Full fine-tune of 421M with Adam ≈ 12-16 bytes/param ≈ **5-7 GB** → fits comfortably on one
  24 GB GPU; the 322M multilingual is lighter still. Proven directly by the official notebook on
  2× 16 GB T4s. LoRA is not really necessary at this scale. Training config: `amp_dtype: bf16`,
  weights stored F16.

---

## Section 2 — Strategic decision: two paths

Your goal — "a model that strongly supports Indonesian, regional languages, English, and their
mixes" — can mean two very different things, with completely different paths:

| | **Path A — Fine-tune Laya (classification)** | **Path B — Alternative generative LLM** |
|---|---|---|
| End result | A fast decision model: intent detection, ticket routing, guardrails, scoring, sentiment classification — in id/regional/en/mixed | A multilingual conversational assistant + code-switching |
| Base | `laya-multilingual` (322M, Apache-2.0) | Qwen3-0.6B/1.7B, Gemma-3-1B, Llama-3.2-1B, **or directly Sahabat-AI 8B/9B & SEA-LION v3.5 (already id+jv+su+en)** |
| Effort | Small-medium (weeks, 1 consumer GPU/Kaggle) | Medium-large (CPT 10-30B tokens + SFT; 1× 4090 3-4 weeks, or multi-GPU cloud 1-2 days) |
| Fundamental limit | **It will never generate text** | Starts from zero on the Laya side (uses none of Laya's pipeline) |

**Recommendation:** if the use case is conversation/generation (an Indonesian-mix chatbot),
choose Path B — retraining Laya will never produce that. If the use case is a fast, cheap
decision/classification layer (even a CPU server — the model is only 322M with one-forward-pass
inference), Path A is very viable and cheap.

---

## Section 3 — Path A: Fine-tuning Laya for Indonesian & regional languages

### 3.1 Mandatory setup before anything else

1. **Use the `laya-multilingual` checkpoint**, not `laya` (root) or `laya-typed-decisions`
   (the latter is English-only per its card).
2. **Patch the router**: set `Router(default="multilingual")` or `lang_guess="id"` — without
   this, plain-ASCII Indonesian text is misrouted to the English checkpoint and accuracy drops
   to 0.318 ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)).
3. **Recalibrate the temperature after training** — the official notebook includes this; the
   Indonesian ECE of 0.305 is still poor, and the root model's evidence shows a temperature
   refit can improve ECE from 0.466 → 0.081.

### 3.2 Labeled data for Indonesian & regional languages (what actually exists)

| Dataset | Contents | Relevance to Laya |
|---|---|---|
| **MASSIVE** (already used in Laya's benchmark) | 51 languages incl. `id` — 60-intent classification, ~12k utterances/lang | Ready-made `choice` format; the main id intent fine-tuning source |
| **NusaCrowd** ([IndoNLP](https://github.com/IndoNLP/nusa-crowd), [arXiv:2212.09648](https://arxiv.org/abs/2212.09648)) | 83 Indonesian NLP datasets + >10 regional languages, uniform loaders | One door to SmSA (sentiment), emotion, CASA, HoASA, etc. → convert to typed-decisions |
| **NusaX-senti / NusaX-MT** ([GitHub](https://github.com/IndoNLP/nusax), [HF](https://huggingface.co/datasets/indonlp/NusaX-MT)) | id, en + 10 regional languages (ace, ban/Balinese, bjn/Banjar, bug, jav/Javanese, mad, min/Minangkabau, nij, sun/Sundanese, bbc/Toba Batak), ~1,000 sentences/lang, native-speaker manual translation, CC-BY-SA 4.0 | The only high-quality regional source for `score` (sentiment) & evaluation |
| **IndoMMLU** ([HF](https://huggingface.co/datasets/indolem/IndoMMLU)) | 14,906 questions, 63 tasks | Knowledge `choice` evaluation |
| **Code-mixed**: annotated 825-row EN-ID tweet dataset (Winata et al. 2018, via [awesome-code-mixing](https://github.com/lingo-iitgn/awesome-code-mixing)); corpus + word-level LID for id-jv-en (Hidayatullah et al. 2023, open access on PMC); EN-ID lexical normalization ([Barik et al. 2019](https://aclanthology.org/D19-5503/)) | Labeled CS data is tiny (hundreds-thousands) | Seed for an internal CS testset + augmentation |
| **IndoRobusta** ([arXiv:2211.05360](https://arxiv.org/abs/2211.05360)) | Code-mixing robustness framework for id + en/sun/jav/mal | Key finding: models tolerate EN-ID mixing far better than ID-regional mixing → regional mixing must be explicitly augmented |
| **NusaBERT** ([arXiv:2403.01652](https://arxiv.org/abs/2403.01652), [HF LazarusNLP](https://huggingface.co/LazarusNLP/NusaBERT-base)) | IndoBERT extended to Indonesian languages & cultures (incl. regional + mixed) | Precedent for an Indonesian-regional encoder; usable as a comparison or teacher |

**Important note**: Laya's original typed-decisions training data is not shared by the
maintainers (issue [#320](https://github.com/NandhaKishorM/laya/issues/320)); the benchmark
splits (1,200 cases / 6,000 decisions) live in the `research/` folder of the GitHub repo. For
Indonesian, the datasets above must be converted yourself into Laya's typed-question schema
(`choice`/`score`/`noul`) — the official fine-tune notebook shows the format.

### 3.3 Training recipe (from the official notebook + precedent)

1. Convert the datasets (Section 3.2) to typed-decisions format: intent classification →
   `choice`; scaled sentiment → `score`; yes/no → `noul`.
2. Run [`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
   as a template: supervised cross-entropy + RLCD, `LR_ENCODER 2.5e-5` / `LR_HEAD 1e-4`,
   4 epochs, fp16 + gradient checkpointing, DDP on 2 GPUs (or 1× 24 GB full fine-tune —
   estimate 5-7 GB for 421M).
3. **Preserve English capability**: mix original EN data (MASSIVE-en, AG News, SST-5 splits
   already used in Laya's benchmark) with the id data — the CPT literature precedent is 10-50%
   source-language replay (Swallow used 10% EN, [arXiv:2404.17790](https://arxiv.org/abs/2404.17790);
   Sahabat-AI ~50% EN/global).
4. **Regional languages**: aggressive oversampling (Sahabat-AI precedent: jv ×3.8, su ×3.8) +
   IndoRobusta-style id-regional CS augmentation.
5. **Code-switching**: build an internal CS testset from real code-mixed tweets
   (Winata/Hidayatullah) — no mature public EN-ID-regional CS benchmark exists.
6. Evaluation: MASSIVE-id (compare against the 0.51 baseline), NusaX-senti 12 languages, a
   homemade CS testset, ECE before/after calibration.

### 3.4 Path A specific risks

- The 256k mmBERT encoder is probably adequate for Latin-script id/jv/su/min — but **fertility
  has never been measured**; measure first (`tokens(text)/word` on id/jv/su samples vs en). If
  fertility is bad, vocabulary extension on a BERT encoder has far weaker precedent than on LLMs —
  consider swapping the backbone to NusaBERT/mBERT and training Laya's head on top.
- Non-Latin regional scripts (Javanese/Balinese script, Bugis Lontara): no data at all; outside
  realistic scope.
- Issue #320 shows the main gap is *encoder comprehension* — so the emphasis belongs on training
  the encoder on target-language text, not just the head (the official notebook trains both).

---

## Section 4 — Corpora & resources for full understanding (applies to both paths)

### 4.1 Indonesian pretraining corpora

| Corpus | Size | Access |
|---|---|---|
| **Indo4B** | ~3.6-4B words, ~23.4 GB, ~250M sentences | [HF SEACrowd/indo4b](https://huggingface.co/datasets/SEACrowd/indo4b); Koto et al. AACL 2020 ([link](https://aclanthology.org/2020.aacl-main.85/)) |
| **mC4-id** | ~22.7B tokens / ~148.3 GB (pre-dedup, Lim et al. 2023) | [HF allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| **CulturaX-id** | `id` subset of 167 languages (6.7T total); exact `id` figure in the [paper's table, arXiv:2309.14050](https://arxiv.org/abs/2309.14050) *(number unverified)* | [HF uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| **OSCAR-id** | LID-filtered CommonCrawl `id` subset | [oscar-corpus.com](https://oscar-corpus.com) |
| **Indonesian Wikipedia** | official snapshots | [HF wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| **SEA-LION Pile (id)** | 27.5B unique tokens used by Sahabat-AI | [Sahabat-AI model card](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) |

Standard cleaning pipeline (CulturaX): fastText LID → per-language KenLM perplexity filter →
exact dedup → MinHash fuzzy dedup → toxicity filtering ([arXiv:2309.14050](https://arxiv.org/abs/2309.14050)).

### 4.2 Regional-language corpora

| Source | Coverage | Notes |
|---|---|---|
| **NusaX** | id, en + 10 regional languages (ace, ban, bjn, bug, jav, mad, min, nij, sun, bbc) | ~1,000 sentences/lang, manual translation, CC-BY-SA 4.0 |
| **NLLB mined bitext** ([HF](https://huggingface.co/datasets/allenai/nllb), [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)) | ace, ban, bjn, bug, jav, min, su — **no mad/nij/bbc** (verified directly in `nllb_lang_pairs.py`) | ~450 GB of LASER3-mined bitext |
| **NusaWrites** ([GitHub](https://github.com/IndoNLP/nusa-writes)) | 12 very-low-resource languages | manual curation + templates |
| **Regional Wikipedias** | jv ~75k articles, su ~60k, min ~230k (many stubs), bjn/ace 10-25k ([official list](https://meta.wikimedia.org/wiki/List_of_Wikipedias)) | jv/su = hundreds of MB to a few GB of text |
| **NusaCrowd** | 83 datasets | the main aggregator |
| **LoraxBench** | 20-Indonesian-language multitask benchmark (2025) | arXiv Aug 2025 (search "LoraxBench") |

**Regional data conclusion**: only jv/su have hundreds-of-MB-to-GB scale data (Sahabat-AI had
only 0.4B unique jv tokens & 0.2B su, then oversampled ×3.8). For ace/mad/bug/bal/bjn/bbc,
natural data is nearly just NusaX (~1k sentences) + NLLB + small Wikipedias → **synthetic
data/back-translation is mandatory**.

### 4.3 EN-ID (and id-regional) code-switching corpora

- Labeled data is tiny: 825 annotated EN-ID tweets (Winata et al. 2018); EN-ID normalization
  pipelines ([Barik et al. 2019](https://aclanthology.org/D19-5503/)); an id-jv-en corpus +
  word-level LID (Hidayatullah et al. 2023, PMC).
- **IndoRobusta** ([arXiv:2211.05360](https://arxiv.org/abs/2211.05360)): models are more robust
  to EN-ID mixing than to ID-regional mixing — a pretraining bias; explicit CS augmentation
  improves robustness.
- **IndoJavE** (2025): a pretrained LM specifically for id-jv-en code-mixed text (search "IndoJavE").
- **LLM-based CS synthesis**: "Conditioning LLMs to Generate Code-Switched Text"
  ([arXiv:2502.12924](https://arxiv.org/abs/2502.12924)) — LLMs without fine-tuning do not
  consistently produce natural CS; **COMMIT** (NAACL 2024, [link](https://aclanthology.org/2024.naacl-long.211/))
  — code-mixed instruction tuning effectively adapts an English-centric LLM to low-resource languages.
- Natural EN-ID mixed documents exist implicitly in CulturaX-id/Indonesian Reddit (minable with
  per-sentence LID).

### 4.4 Instruction/SFT corpora (for Path B)

| Dataset | Contents | Source |
|---|---|---|
| **Aya Collection** | 513M instances, >100 languages incl. **jv & su** | [HF CohereForAI](https://huggingface.co/datasets/CohereForAI/aya_collection) |
| **Cendol Collection** | instructions, 23 tasks × 10 languages (Indonesian + regional) | [GitHub IndoNLP/cendol](https://github.com/IndoNLP/cendol) |
| **Bactrian-X id** | ~67k id instruction pairs (machine-translated) | [GitHub mbzuai-nlp](https://github.com/mbzuai-nlp/bactrian-x) |
| **Sahabat-AI instruct** | ~448k id pairs | [model card](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-instruct) |
| **OASST1 id** | small subset | [HF](https://huggingface.co/datasets/OpenAssistant/oasst1) |

### 4.5 Evaluation benchmarks

- **IndoMMLU** (14,906 questions), **Nusantara/NusaNLU** ([arXiv:2212.09648](https://arxiv.org/abs/2212.09648)),
  **SEA HELM/BHASA** ([arXiv:2503.02361](https://arxiv.org/abs/2503.02361),
  [leaderboard](https://aisingapore.github.io/seahelm-bhasa-leaderboard/)), **IndoMTEB**
  ([LazarusNLP](https://github.com/LazarusNLP); part of MMTEB [arXiv:2502.13443](https://arxiv.org/abs/2502.13443)),
  **IndoCareer** ([arXiv:2409.08564](https://arxiv.org/abs/2409.08564)), the
  [indonesian-nlp](https://huggingface.co/spaces/indonesian-nlp/open-indonesian-llm-leaderboard)
  & [Sahabat-AI](https://huggingface.co/spaces/Sahabat-AI/Sahabat-AI-Leaderboard) leaderboards.
- **Code-switching: no mature EN-ID-regional CS benchmark exists** — a real gap; build an
  internal eval (IndoRobusta-style + Winata/Hidayatullah samples).

---

## Section 5 — Path B: Alternative generative model (language adaptation playbook)

> Applies if the end goal is a conversational/generative model for id + regional + en + CS.
> Laya is not used at all on this path.

### 5.1 Base selection — check first whether retraining is needed at all

| Base | Why | License |
|---|---|---|
| **Sahabat-AI 8B/9B (instruct)** | **Already** continued-pretrained on id + jv + su + en by GoTo/Indosat — supports id, jv, su, bal, batak + en; may just need CS fine-tuning | [HF GoToCompany](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-instruct) (Llama 3 Community License) |
| **SEA-LION v3.5** | CPT on ~200B tokens of Southeast Asian data over Gemma-2 9B / Llama-3.1 8B | [GitHub aisingapore/sealion](https://github.com/aisingapore/sealion), [arXiv:2504.05747](https://arxiv.org/abs/2504.05747) |
| **Qwen3-0.6B/1.7B / Gemma-3-1B / Llama-3.2-1B** | If a small ~1B class is genuinely needed and full self-adaptation is wanted | Qwen (Apache-2.0 on some variants), Gemma Terms, Llama Community License |

### 5.2 Prior-art precedent (precise numbers you can copy)

| Model | Base | Data | Key technique | Source |
|---|---|---|---|---|
| **Sahabat-AI v1** | Llama-3-8B (CPT) | 50B tokens: id 27.5B + EN/global (Dolma) + jv 0.4B×3.8 + su 0.2B×3.8 | **No vocab extension** (Llama-3's 128k tokenizer suffices); LR 1e-5, batch 256, bf16, 32×H100 ~5 days; SFT on 448k id instructions | [model card](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) |
| **SEA-LION v3/v3.5** | Gemma-2 9B (CPT ~200B tokens) | SEALD corpus | per-language SEA token fertility analysis driving tokenizer decisions | [arXiv:2504.05747](https://arxiv.org/abs/2504.05747) |
| **Komodo-7B** | Llama-2-7B | only 8.5B tokens of id+regional | **vocabulary expansion** + incremental pretraining; supports 11 regional languages | [HF](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base), [blog](https://tech.yellow.ai/komodo-7b-the-first-llm-for-regional-languages-in-indonesia-2a5256d09ffa) |
| **Swallow (Japan)** | Llama-2 7-70B | 20-100B ja tokens + 10% en replay | vocab extension +20% with **mean-init embeddings** → 56% token reduction, 78% faster generation; largest gains in the first 20B tokens; EN drops 2-5 points | [arXiv:2404.17790](https://arxiv.org/abs/2404.17790) |
| **Chinese-LLaMA-Alpaca** | Llama | +20k zh tokens (32k → 49,953) | mean-init of new embeddings; **two-stage LoRA CPT** (2.97% → 6.06% of parameters) | [arXiv:2304.08177](https://arxiv.org/abs/2304.08177), [training wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Training-Details) |
| **Bakpia-V1-0.5B-Javanese** | small open model | Javanese data | Javanese-specific instruct tuning — proof of feasibility at the 0.5B class | [HF](https://huggingface.co/afrizalha/Bakpia-V1-0.5B-Javanese) |

*(Note: "Nusantara (GoTo)", "Merdeka", "Merah Putih", "Bhasa", "Aquila" — no credible technical
documentation or open weights found; do not use as references. Garuda (Indosat-Tech Mahindra,
1.2B params, 16B tokens) has no verifiable open weights.)*

### 5.3 End-to-end recipe for a ~1B generative model

1. **Tokenizer**: measure id/jv/su fertility on the base. If the base is Latin-friendly
   (Qwen 151k / Gemma / Llama-3 128k) → **keep the vocab** (Sahabat-AI precedent). Extend only
   if target fertility > ~1.5-2× the source language and you have ≥ several billion tokens;
   every ~10k new tokens ≈ +20M parameters on a 1B model that must learn from scratch
   (extension precedents: Chinese-LLaMA/Swallow/Komodo, always mean-init).
2. **CPT full fine-tune bf16**: 10-30B tokens (biggest gains in the first 20B — the Swallow
   curve). Mix: **55-70% Indonesian** (CulturaX-id + Indo4B + Wiki/News) + **10-30% EN replay**
   (Dolma/C4) + **5-15% regional** (jv/su oversampled ×3-4; others via NLLB mining + Wikipedia +
   NusaWrites) + **2-5% natural EN-ID CS documents** (mined from the web). Precedent
   hyperparameters: LR 5e-5-2e-4 (1e-4@7B Swallow; 1e-5@8B Sahabat-AI), ~1k-step warmup, cosine
   decay, ~0.5-1M-token global batch, AdamW β=(0.9, 0.95), wd 0.1, FlashAttention-2.
3. **SFT**: Aya (id/jav/su) + Cendol + Bactrian-X id + OASST id + **synthetic code-mixed
   instructions** (the [COMMIT](https://aclanthology.org/2024.naacl-long.211/) &
   [arXiv:2502.12924](https://arxiv.org/abs/2502.12924) recipes) + EN instructions (to preserve
   EN instruct ability). LoRA/QLoRA is usually enough at this stage.
4. **Evaluation**: IndoMMLU + NusaNLU + SEA HELM (id/jv/su) + IndoMTEB + a light EN eval +
   **your own CS eval** (IndoRobusta-style).
5. **EN forgetting mitigation**: 10-30% EN replay; LoRA "learns less, forgets less"
   ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)); post-training model merging
   ([NVIDIA survey](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)).

### 5.4 Compute estimate (~1B model, 20B tokens, full FT ≈ 1.2×10²⁰ FLOPs; calibrated on the [TinyLlama](https://arxiv.org/abs/2402.17777) datapoint)

| Hardware | Estimated time |
|---|---|
| 1× A100 80GB | ~7-11 days (40-50% MFU) |
| 8× A100 | ~1-1.5 days |
| 1× RTX 4090 24GB | ~3-4 weeks |
| SFT/QLoRA (1-5B tokens) on 1× 24GB | hours-2 days |

Full-FT VRAM for 1B: bf16 weights ~2GB + optimizer ~12-16GB + activations (grad ckpt) ~4GB ≈
18-22GB → **one 24 GB consumer GPU suffices** (8-bit optimizer if tight). Tools:
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (one-stop, GUI),
[torchtune](https://github.com/pytorch/torchtune) (native, supports vocab resize),
[Unsloth](https://github.com/unslothai/unsloth) (fast SFT/LoRA),
[axolotl](https://github.com/axolotl-ai-cloud/axolotl), HF transformers+accelerate.

**Scratch vs CPT vs LoRA**: a ~1B from scratch needs ~1-3T tokens (TinyLlama: 3T tokens,
16×A100, 90 days) — uneconomical; full CPT = the proven default; LoRA-CPT = the budget option
with a lower ceiling for languages very different from the base.

---

## Section 6 — Risks, licensing, & what was not found

| Risk | Facts & mitigation |
|---|---|
| **Fundamental misunderstanding** | Laya is not generative — be sure of the use case before investing. |
| **Licensing** | Laya: Apache-2.0 (safe, commercial OK). Datasets: **NusaX CC-BY-SA 4.0 (share-alike!)**, CulturaX/mC4 bound by Common Crawl policy. Generative bases: Llama 3 Community License (>700M MAU needs permission), Gemma Terms, Qwen partially Apache-2.0. |
| **EN degradation** | Documented 2-5 points with 10% replay (Swallow); 10-30% replay + LoRA/merging. |
| **Tiny regional data** | jv/su only 0.2-0.4B unique tokens; others ~1k sentences → oversample ×3-4, NLLB back-translation, NusaWrites curation, LLM synthesis. |
| **Immature CS benchmarks** | Build an internal CS eval + human eval. |
| **Undocumented tokenizer fertility** (Laya-specific) | Measure before making any vocab decision. |
| **"Ghost models"** | Garuda/Merdeka/Merah Putih/Bhasa/Nusantara-GoTo are undocumented — do not use as technical references. |

**Not found / unverified** (honest from the research): mmBERT provenance (later found in config:
`jhu-clsp/mmBERT-base`); Laya's training corpus; exact CulturaX-id token count; arXiv IDs of
IndoMTEB & CodeMixBench (titles exist, stable links unverified); "SANME/SanmeTalks/IndoVoA"
datasets (likely wrong names); "SALSA"/"UMBC" as EN-ID CS datasets (what exists is IndoCollex &
the alay dictionary — about slang, not CS).

---

## Section 7 — Recommended next steps

1. **Decide the use case**: a decision/classification layer (→ Path A) or
   conversation/generation (→ Path B). This decision determines everything.
2. **Cheap Path A experiment (1-2 days)**: pull `laya-multilingual` → patch the router →
   evaluate the baseline on MASSIVE-id + NusaX-senti → fine-tune with the Kaggle notebook on
   MASSIVE-id → measure the delta.
3. **Cheap Path B experiment**: try Sahabat-AI/SEA-LION instruct with CS prompts first — perhaps
   80% of the need is met without any training; only then decide on CPT.
4. **Measure token fertility** for id/jv/su/en on any candidate tokenizer before training
   (SEA-LION methodology).
5. Collect & convert data into the target format (typed-decisions for A; CPT/SFT mix for B) —
   see Sections 3.2 & 4.

---

## Appendix — All primary sources

**Laya**: [HF laya](https://huggingface.co/convaiinnovations/laya) ·
[HF laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) ·
[HF laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) ·
[GitHub NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) ·
[fine-tune notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) ·
[51-language sweep results](https://raw.githubusercontent.com/NandhaKishorM/laya/research/research/results/cpu_51_language_sweep.json) ·
[PR #286 router](https://github.com/NandhaKishorM/laya/pull/286) ·
[issue #320](https://github.com/NandhaKishorM/laya/issues/320) ·
[laya.convaiinnovations.com](https://laya.convaiinnovations.com)

**Prior art & techniques**: [Sahabat-AI](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) ·
[SEA-LION](https://arxiv.org/abs/2504.05747) · [Swallow](https://arxiv.org/abs/2404.17790) ·
[Chinese-LLaMA-Alpaca](https://arxiv.org/abs/2304.08177) ·
[Komodo-7B](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base) ·
[LoRA learns less and forgets less](https://arxiv.org/abs/2405.09673) ·
[QLoRA](https://arxiv.org/abs/2305.14314) ·
[TinyLlama](https://arxiv.org/abs/2402.17777) ·
[CulturaX](https://arxiv.org/abs/2309.14050) ·
[NusaBERT](https://arxiv.org/abs/2403.01652) ·
[NusaMT-7B](https://arxiv.org/abs/2410.10461)

**Indonesian/regional/CS data**: [Indo4B](https://huggingface.co/datasets/SEACrowd/indo4b) ·
[NusaX](https://github.com/IndoNLP/nusax) ·
[NusaCrowd](https://github.com/IndoNLP/nusa-crowd) ·
[NusaWrites](https://github.com/IndoNLP/nusa-writes) ·
[Cendol](https://github.com/IndoNLP/cendol) ·
[NLLB](https://huggingface.co/datasets/allenai/nllb) ·
[Aya Collection](https://huggingface.co/datasets/CohereForAI/aya_collection) ·
[Bactrian-X](https://github.com/mbzuai-nlp/bactrian-x) ·
[IndoRobusta](https://arxiv.org/abs/2211.05360) ·
[COMMIT](https://aclanthology.org/2024.naacl-long.211/) ·
[Conditioning LLMs for CS](https://arxiv.org/abs/2502.12924) ·
[IndoMMLU](https://huggingface.co/datasets/indolem/IndoMMLU) ·
[SEA HELM](https://arxiv.org/abs/2503.02361) ·
[IndoCollex/normalization](https://github.com/NormalisasiKata/NormalisasiKata) ·
[awesome-code-mixing](https://github.com/lingo-iitgn/awesome-code-mixing)
