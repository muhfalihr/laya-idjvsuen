# Riset Mendalam: Model Laya & Jalur Retraining untuk Bahasa Indonesia, Bahasa Daerah, Inggris, dan Code-Switching

**Bahasa**: **Indonesia** | [English](RISET-LAYA.en.md)

> Disusun 24 September 2026. Semua klaim bersumber primer (HuggingFace, GitHub, arXiv, ACL Anthology) kecuali yang ditandai eksplisit "tidak terverifikasi". Riset oleh 2 agen riset paralel, dikonsolidasikan dalam dokumen ini.

---

## Ringkasan Eksekutif (baca ini dulu)

1. **⚠️ Laya BUKAN model generatif/percakapan.** Laya adalah *decision model* non-autoregressive berbasis encoder BERT untuk klasifikasi terstruktur: menerima state (teks/ticket/JSON) + pertanyaan bertipe (`choice`, `score`, `noul`) lalu mengembalikan jawaban terketik + probabilitas terkalibrasi dalam satu forward pass. Model card-nya sendiri menyatakan: *"never generates text, so there is nothing to parse and nothing to hallucinate"* ([README resmi](https://huggingface.co/convaiinnovations/laya/raw/main/README.md)). **Tidak ada komponen vision/audio sama sekali.**
2. **"Retraining agar support bahasa Indonesia" pada Laya hanya bisa berarti** meningkatkan kemampuan *klasifikasi/keputusan* berbahasa Indonesia — bukan menjadikannya chatbot. Jika tujuan Anda adalah model percakapan multibahasa (id + daerah + en + campuran), basis yang tepat adalah LLM generatif lain (mis. Qwen/Gemma, atau langsung Sahabat-AI / SEA-LION yang sudah support id+jv+su+en) — playbook lengkapnya ada di [Bagian 5](#bagian-5--jalur-b-model-generatif-alternatif-playbook-adaptasi-bahasa).
3. **Kondisi Indonesia saat ini belum layak produksi:** akurasi checkpoint multilingual di MASSIVE (51 bahasa, 20 opsi) hanya **0,51** (macro-F1 0,4955, ECE 0,305); router bahasanya salah mendeteksi teks Indonesia Latin sebagai Inggris ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)). Ini justru berarti ruang perbaikan lewat fine-tune berbahasa Indonesia sangat besar.
4. **Lisensi Apache-2.0** — retraining, distribusi bobot turunan, dan penggunaan komersial diizinkan penuh. Pipeline fine-tune resmi open source dan terbukti jalan di **2× T4 Kaggle (gratis)**.
5. **Kemungkinan tertukar nama**: ada "LAYA" lain milik Articul8 AI (model Sanskrit 32B, perusahaan berbeda — [Hindustan Times](https://www.hindustantimes.com/india-news/with-laya-sol-articul8-puts-language-scholars-at-heart-of-heritage-ai-101790155154789.html)). Jangan campuradukkan.

---

## Bagian 1 — Apa itu Laya sebenarnya

### 1.1 Definisi & keluarga checkpoint

| Checkpoint | Backbone | Parameter | Context | Bahasa |
|---|---|---|---|---|
| `laya` (root) | ModernBERT-large + decision head | 421M | 512 (config 8192 tokenizer) | Inggris |
| `laya-multilingual` | "mmBERT-base" (bidirectional, 22 layer, hidden 768, vocab 256k) | 322M | 1024 (up to 8.192) | 51+ bahasa, termasuk `id`, `ms`, `jv` |
| `laya-typed-decisions` | fine-tune dari `laya` | ~421M | — | *"English only. Use laya-multilingual for other languages."* |

Sumber: [model card laya](https://huggingface.co/convaiinnovations/laya), [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions).

- Base model terkonfirmasi dari file config: root memakai `answerdotai/ModernBERT-large` ([rl_agent_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/rl_agent_config.json)); arsitektur encoder `ModernBertForMaskedLM`, vocab 50.368, 28 layer, hidden 1024 ([encoder/config.json](https://huggingface.co/convaiinnovations/laya/raw/main/encoder/config.json)). Multilingual: `ModernBertForMaskedLM`/`modernbert`, vocab 256.000, 22 layer, hidden 768 ([config multilingual](https://huggingface.co/convaiinnovations/laya-multilingual/raw/main/encoder/config.json)).
- `pipeline_tag: text-classification`; text-only — tidak ada `vision_config`/`audio_config` di config mana pun ([HF API](https://huggingface.co/api/models/convaiinnovations/laya)).
- Provenance "mmBERT-base" **tidak dijelaskan di sumber publik mana pun** (tidak ada repo/paper asal yang bisa dilacak).
- Model sangat baru: dibuat 18 Sep 2026, terakhir diubah 24 Sep 2026 ([HF API](https://huggingface.co/api/models/convaiinnovations/laya)).

### 1.2 Tokenizer — implikasi untuk bahasa Indonesia

- **Root (Inggris)**: `PreTrainedTokenizerFast` gaya WordPiece ModernBERT, vocab 50.368, `[CLS]/[MASK]/[SEP]/[PAD]/[UNK]` — pada dasarnya *English-centric* ([tokenizer_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/tokenizer/tokenizer_config.json)).
- **Multilingual**: `PreTrainedTokenizerFast`, vocab **256.000**, dengan special token gaya Gemma (`<bos>/<eos>`, `start_of_turn/end_of_turn`) ([tokenizer_config multilingual](https://huggingface.co/convaiinnovations/laya-multilingual/raw/main/tokenizer/tokenizer_config.json)).
- Bahasa `id`, `ms`, `jv` terdaftar resmi di tag bahasa HF multilingual ([HF API laya-multilingual](https://huggingface.co/api/models/convaiinnovations/laya-multilingual)).
- **Efisiensi token-per-kata (token fertility) untuk bahasa Indonesia/daerah tidak pernah diukur/didokumentasikan publik.** Vocab 256k pola Gemma kemungkinan besar meng-cover teks Latin Indonesia dengan cukup baik, tapi ini hipotesis — **wajib diukur sendiri** (metodologi fertility dari [paper SEA-LION, arXiv:2504.05747](https://arxiv.org/abs/2504.05747)).
- Untuk aksara non-Latin (mis. aksara Jawa `java`/`Javanese` script): tidak ada data sama sekali.

### 1.3 Metode training & data asli

- Metode: **RLCD** — *"reinforcement learning against strictly proper scoring rules"*, REINFORCE dengan group-mean baseline (gaya GRPO), reward log + spherical + ranked probability score, TD(λ=1.0) untuk multi-turn ([README](https://huggingface.co/convaiinnovations/laya/raw/main/README.md)).
- Metadata training root: 7.313 updates, 1 epoch, 1,96 jam, `world_size: 1`, `fine_tuned_from_checkpoint: true` ([rl_agent_config.json](https://huggingface.co/convaiinnovations/laya/raw/main/rl_agent_config.json)).
- **Korpus training TIDAK diungkap.** Model card hanya merujuk split benchmark: typed-decisions (1.200 kasus / 6.000 keputusan), MASSIVE, XNLI, AG News, SST-5, Banking77, CLINC150, DAIR Emotion. Issue [#320](https://github.com/NandhaKishorM/laya/issues/320) masih meminta maintainer membagikan data training typed-decisions (belum dijawab).

### 1.4 Kode & infrastruktur

- **Repo GitHub**: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — Apache-2.0, 21.128 stars, 1.800 fork, sangat aktif (push terakhir 24 Sep 2026) ([API GitHub](https://api.github.com/repos/NandhaKishorM/laya)). Isi: SDK Python `laya/`, `laya-ts/`, `research/` (data benchmark), `notebooks/`, `benchmarks/`, `scripts/`, `examples/`, `docker/`, `docs/`.
- **Notebook fine-tune resmi**: [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) — RLCD + supervised cross-entropy, DDP `torchrun` di **2× T4 Kaggle (16 GB/GPU)**, fp16 + gradient checkpointing, `LR_ENCODER 2.5e-5` / `LR_HEAD 1e-4`, 4 epoch, melatih encoder + head sekaligus, plus **kalibrasi temperatur pasca-training**.
- **Pretraining dari nol: tidak ada jalurnya** di sumber publik. Yang open source hanya fine-tuning dari checkpoint.
- Format: safetensors; root repo total 2,37 GB (3 checkpoint di subfolder), multilingual 678 MB ([tree](https://huggingface.co/convaiinnovations/laya/tree/main)). **Tidak ada GGUF resmi** (26 quantization komunitas disebut di card). Tidak didukung vLLM/llama.cpp (bukan causal LM). Paket: `pip install laya` dengan extras `[serve]` (FastAPI), `[mcp]`, `[langchain]`, `[onnx]`, `[fast]`.
- Paper khusus Laya: **tidak ada**. Situs resmi merujuk 2 arXiv paper pendiri ([2503.23303](https://arxiv.org/abs/2503.23303), [2510.01237](https://arxiv.org/abs/2510.01237)) yang tidak menyebut Laya/ConvAI. Halaman produk: [laya.convaiinnovations.com](https://laya.convaiinnovations.com).

### 1.5 Bukti empiris kondisi bahasa Indonesia (data primer)

Sweep MASSIVE 51 bahasa × 20 opsi ([hasil riset repo, `research/results/cpu_51_language_sweep.json`](https://raw.githubusercontent.com/NandhaKishorM/laya/research/research/results/cpu_51_language_sweep.json)):

| Checkpoint | Akurasi id | ECE | Catatan |
|---|---|---|---|
| `laya` (root Inggris) | **0,36** | 0,613 | mean confidence 0,962 — sangat overconfident |
| `laya-multilingual` | **0,51** | 0,305 | macro-F1 0,4955; akurasi@50% coverage 0,72 |

Pembanding pada multilingual: en 0,68; ja 0,64; terlemah Amharic 0,11.

- **Bug router untuk bahasa Latin Indonesia** ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)): *"Indonesian reaches 0.318 instead of 0.409, because nothing in plain-ASCII Indonesian tells it apart from English"* — kata `di` terdeteksi sebagai Italia; id/ms/jv hanya mengandalkan "stray list hits". **Workaround wajib: `Router(default="multilingual")` atau `lang_guess="id"`.** Code-switching Latin (id-en-jv) persis di titik lemah router berbasis script; mixed-script non-Latin hanya ditangani via deteksi kehadiran script ([PR #122](https://github.com/NandhaKishorM/laya/pull/122)).
- **Bukti gap ada di encoder, bukan head** ([issue #320](https://github.com/NandhaKishorM/laya/issues/320)): untuk Spanyol, Laya 60,8% vs Qwen-4B 84,3%; saat prompt dialihbahasakan ke Inggris jadi 76,5% vs 74,5% — *"laya's gap on these prompts is mostly the encoder's non-English comprehension, not the typed-decision heads"*. Artinya: fine-tune encoder pada data bahasa target adalah target intervensi yang tepat.
- Limitasi lain dari model card: base near-chance zero-shot typed-decisions (0,362); overconfidence (ECE 0,466 → 0,081 setelah refit temperatur); `noul` bisa anchor ke label ([#156](https://github.com/NandhaKishorM/laya/issues/156)); `act_probability` tanpa sinyal ([#185](https://github.com/NandhaKishorM/laya/issues/185)); position bias pada `score` multilingual ([#131](https://github.com/NandhaKishorM/laya/issues/131)).

### 1.6 Lisensi & komunitas

- **Apache-2.0** — terkonfirmasi di [HF API](https://huggingface.co/api/models/convaiinnovations/laya) (kedua repo), GitHub, dan situs produk (*"100% open-source Apache 2.0 weights"*). Tidak ada klausa non-komersial/gated.
- HF: `laya` 3.168 likes; GitHub: 21.128 stars. Klaim di card: 42 finetunes, 26 quantizations, 32 Spaces turunan.
- Issue relevan bahasa: [#320](https://github.com/NandhaKishorM/laya/issues/320) (minta data training & checkpoint multilingual typed-decisions), [#286](https://github.com/NandhaKishorM/laya/pull/286) (routing id/ms/jv), #42, #202, #207 (Portugis), #127 (kalibrasi Mandarin), #113, #122 (mixed-script).

### 1.7 Estimasi memori training (hitungan standar dari jumlah parameter)

- Full fine-tune 421M dengan Adam ≈ 12–16 byte/param ≈ **5–7 GB** → muat di 1× GPU 24 GB dengan nyaman; 322M multilingual lebih ringan lagi. Dibuktikan langsung oleh notebook resmi di 2× T4 16 GB. LoRA tidak benar-benar diperlukan pada skala ini. Config training: `amp_dtype: bf16`, bobot disimpan F16.

---

## Bagian 2 — Keputusan strategis: dua jalur

Tujuan Anda — "model sangat support bahasa Indonesia, bahasa daerah, Inggris, dan campurannya" — bisa berarti dua hal berbeda, dan jalurnya berbeda total:

| | **Jalur A — Fine-tune Laya (klasifikasi)** | **Jalur B — LLM generatif alternatif** |
|---|---|---|
| Hasil akhir | Model keputusan cepat: intent detection, routing tiket, guardrail, scoring, klasifikasi sentimen — dalam id/daerah/en/campuran | Chatbot/asisten percakapan multibahasa + code-switching |
| Basis | `laya-multilingual` (322M, Apache-2.0) | Qwen3-0.6B/1.7B, Gemma-3-1B, Llama-3.2-1B, **atau langsung Sahabat-AI 8B/9B & SEA-LION v3.5 (sudah id+jv+su+en)** |
| Effort | Kecil–sedang (mingguan, 1 GPU konsumen/Kaggle) | Sedang–besar (CPT 10–30B token + SFT; 1× 4090 3–4 minggu, atau cloud multi-GPU 1–2 hari) |
| Batasan fundamental | **Tidak akan pernah menghasilkan teks** | Mulai dari nol di sisi infra Laya (tidak pakai pipeline Laya sama sekali) |

**Rekomendasi:** jika use case Anda sebenarnya adalah percakapan/generasi (chatbot bahasa Indonesia-campuran), pilih Jalur B — retraining Laya tidak akan pernah menghasilkan itu. Jika use case-nya lapisan keputusan/klasifikasi cepat dan murah (server CPU pun bisa, model cuma 322M dengan inference satu forward pass), Jalur A sangat layak dan murah.

---

## Bagian 3 — Jalur A: Fine-tune Laya untuk bahasa Indonesia & daerah

### 3.1 Setup wajib sebelum apa pun

1. **Gunakan checkpoint `laya-multilingual`, bukan `laya` (root) atau `laya-typed-decisions`** (yang terakhir English-only sesuai card-nya).
2. **Patch router**: set `Router(default="multilingual")` atau `lang_guess="id"` — tanpa ini, teks Indonesia murni ASCII salah diroute ke checkpoint Inggris dan akurasi jatuh ke 0,318 ([PR #286](https://github.com/NandhaKishorM/laya/pull/286)).
3. **Kalibrasi ulang temperatur setelah training** — notebook resmi sudah menyertakannya; ECE Indonesia 0,305 masih buruk, dan bukti root model menunjukkan refit temperatur bisa memperbaiki ECE dari 0,466 → 0,081.

### 3.2 Data berlabel untuk bahasa Indonesia & daerah (yang benar-benar tersedia)

| Dataset | Isi | Relevansi untuk Laya |
|---|---|---|
| **MASSIVE** (sudah dipakai benchmark Laya) | 51 bahasa termasuk `id` — intent classification 60 intent, ~12k utterance/bahasa | Format `choice` siap pakai; sumber utama fine-tune intent id |
| **NusaCrowd** ([IndoNLP](https://github.com/IndoNLP/nusa-crowd), [arXiv:2212.09648](https://arxiv.org/abs/2212.09648)) | 83 dataset NLP Indonesia + >10 bahasa daerah, loader seragam | Satu pintu untuk SmSA (sentimen), emosi, CASA, HoASA, dll. → konversi ke format typed-decisions |
| **NusaX-senti / NusaX-MT** ([GitHub](https://github.com/IndoNLP/nusax), [HF](https://huggingface.co/datasets/indonlp/NusaX-MT)) | id, en + 10 bahasa daerah (ace, ban/Bali, bjn/Banjar, bug, jav/Jawa, mad, min/Minang, nij, sun/Sunda, bbc/Batak), ~1.000 kalimat/bahasa, terjemahan manual penutur asli, CC-BY-SA 4.0 | Satu-satunya sumber daerah berkualitas tinggi untuk `score` (sentimen) & evaluasi |
| **IndoMMLU** ([HF](https://huggingface.co/datasets/indolem/IndoMMLU)) | 14.906 soal, 63 tugas | Evaluasi `choice` pengetahuan |
| **Code-mixed**: dataset tweet EN-ID 825 baris beranotasi (Winata et al. 2018, via [awesome-code-mixing](https://github.com/lingo-iitgn/awesome-code-mixing)); korpus + word-level LID id-jv-en (Hidayatullah et al. 2023, open access di PMC); normalisasi leksikal EN-ID ([Barik et al. 2019](https://aclanthology.org/D19-5503/)) | Data CS berlabel sangat kecil (ratusan–ribuan) | Benih untuk testset CS internal + augmentasi |
| **IndoRobusta** ([arXiv:2211.05360](https://arxiv.org/abs/2211.05360)) | Framework robustness code-mixing id + en/sun/jav/mal | Temuan kunci: model lebih tahan mixing EN-ID daripada ID-daerah → mixing daerah harus di-augmentasi eksplisit |
| **NusaBERT** ([arXiv:2403.01652](https://arxiv.org/abs/2403.01652), [HF LazarusNLP](https://huggingface.co/LazarusNLP/NusaBERT-base)) | IndoBERT di-extend ke bahasa & budaya Indonesia (termasuk daerah + campuran) | Preseden encoder bahasa Indonesia-daerah; bisa dipakai sebagai pembanding atau teacher |

**Catatan penting**: data training typed-decisions asli Laya tidak dibagikan maintainer (issue [#320](https://github.com/NandhaKishorM/laya/issues/320)); split benchmark-nya (1.200 kasus / 6.000 keputusan) ada di folder `research/` repo GitHub. Untuk bahasa Indonesia, dataset di atas perlu dikonversi sendiri ke skema pertanyaan-typed Laya (`choice`/`score`/`noul`) — notebook fine-tune resmi menunjukkan formatnya.

### 3.3 Resep training (dari notebook resmi + preseden)

1. Konversi dataset (Bagian 3.2) ke format typed-decisions: klasifikasi intent → `choice`; sentimen skala → `score`; ya/tidak → `noul`.
2. Jalankan [`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) sebagai template: supervised cross-entropy + RLCD, `LR_ENCODER 2.5e-5` / `LR_HEAD 1e-4`, 4 epoch, fp16 + gradient checkpointing, DDP 2 GPU (atau 1× 24 GB full fine-tune — estimasi 5–7 GB untuk 421M).
3. **Jaga kemampuan Inggris**: campur data EN asli (MASSIVE-en, AG News, SST-5 split yang sudah dipakai benchmark Laya) dengan data id — preseden replay 10–50% bahasa sumber dari literatur CPT (Swallow memakai 10% EN, [arXiv:2404.17790](https://arxiv.org/abs/2404.17790); Sahabat-AI ~50% EN/global).
4. **Bahasa daerah**: oversample agresif (preseden Sahabat-AI: jv ×3,8, su ×3,8) + augmentasi CS id-daerah ala IndoRobusta.
5. **Code-switching**: buat testset CS internal dari tweet code-mixed asli (Winata/Hidayatullah) — benchmark publik CS EN-ID-daerah belum matang.
6. Evaluasi: MASSIVE-id (bandingkan angka dasar 0,51), NusaX-senti 12 bahasa, testset CS buatan, ECE sebelum/sesudah kalibrasi.

### 3.4 Risiko spesifik Jalur A

- Encoder mmBERT 256k kemungkinan cukup untuk Latin id/jv/su/min — tapi **belum ada pengukuran fertility**; ukur dulu (`token(text)/kata` pada sampel id/jv/su vs en). Jika fertility buruk, opsi vocab-extension pada encoder BERT jauh lebih jarang dipresedentkan daripada pada LLM — pertimbangkan ganti backbone ke NusaBERT/mBERT lalu latih head Laya di atasnya.
- Bahasa daerah non-Latin (aksara Jawa/Bali/bugis Lontara): tanpa data apapun; keluar scope realistis.
- Issue #320 menunjukkan gap utama ada di *encoder comprehension* — berarti yang harus ditekankan adalah training encoder pada teks bahasa target, bukan sekadar head (notebook resmi memang melatih keduanya).

---

## Bagian 4 — Korpus & sumber daya untuk pemahaman menyeluruh (berlaku kedua jalur)

### 4.1 Korpus pretraining bahasa Indonesia

| Korpus | Ukuran | Akses |
|---|---|---|
| **Indo4B** | ~3,6–4 M kata, ~23,4 GB, ~250 juta kalimat | [HF SEACrowd/indo4b](https://huggingface.co/datasets/SEACrowd/indo4b); paper Koto et al. AACL 2020 ([link](https://aclanthology.org/2020.aacl-main.85/)) |
| **mC4-id** | ~22,7 M token / ~148,3 GB (pre-dedup, Lim et al. 2023) | [HF allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| **CulturaX-id** | subset `id` dari 167 bahasa (total 6,7T token); angka pasti `id` ada di tabel [paper arXiv:2309.14050](https://arxiv.org/abs/2309.14050) *(tidak diverifikasi angkanya)* | [HF uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| **OSCAR-id** | subset `id` CommonCrawl ber-LID | [oscar-corpus.com](https://oscar-corpus.com) |
| **Wikipedia id** | snapshot resmi | [HF wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) |
| **SEA-LION Pile (id)** | 27,5 M token unik dipakai Sahabat-AI | [model card Sahabat-AI](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) |

Pipeline cleaning standar (CulturaX): fastText LID → KenLM perplexity filter per bahasa → exact dedup → MinHash fuzzy dedup → filter toksik ([arXiv:2309.14050](https://arxiv.org/abs/2309.14050)).

### 4.2 Korpus bahasa daerah

| Sumber | Cakupan | Catatan |
|---|---|---|
| **NusaX** | id, en + 10 bahasa daerah (ace, ban, bjn, bug, jav, mad, min, nij, sun, bbc) | ~1.000 kalimat/bahasa, terjemahan manual, CC-BY-SA 4.0 |
| **NLLB mined bitext** ([HF](https://huggingface.co/datasets/allenai/nllb), [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)) | ace, ban, bjn, bug, jav, min, su — **tidak mencakup mad/nij/bbc** (diverifikasi langsung di `nllb_lang_pairs.py`) | ~450 GB bitext hasil mining LASER3 |
| **NusaWrites** ([GitHub](https://github.com/IndoNLP/nusa-writes)) | 12 bahasa sangat low-resource | kurasi manual + template |
| **Wikipedia daerah** | jv ~75k artikel, su ~60k, min ~230k (banyak stub), bjn/ace 10–25k ([daftar resmi](https://meta.wikimedia.org/wiki/List_of_Wikipedias)) | jv/su = ratusan MB–beberapa GB teks |
| **NusaCrowd** | 83 dataset | agregator utama |
| **LoraxBench** | benchmark multitask 20 bahasa Indonesia (2025) | arXiv Agustus 2025 (cari "LoraxBench") |

**Kesimpulan data daerah**: hanya jv/su yang punya data skala ratusan MB–GB (Sahabat-AI hanya punya 0,4B token unik jv & 0,2B su, lalu oversample ×3,8). Untuk ace/mad/bug/bal/bjn/bbc, data alami nyaris hanya NusaX (~1k kalimat) + NLLB + Wikipedia kecil → **synthetic data/back-translation wajib**.

### 4.3 Korpus code-switching EN–ID (dan id–daerah)

- Data berlabel sangat kecil: tweet EN-ID 825 beranotasi (Winata et al. 2018); pipeline normalisasi EN-ID ([Barik et al. 2019](https://aclanthology.org/D19-5503/)); korpus id-jv-en + word-level LID (Hidayatullah et al. 2023, PMC).
- **IndoRobusta** ([arXiv:2211.05360](https://arxiv.org/abs/2211.05360)): model lebih robust terhadap mixing EN-ID daripada ID-daerah — bias pretraining; augmentasi CS eksplisit meningkatkan robustness.
- **IndoJavE** (2025): LM pre-trained khusus code-mixed id-jv-en (cari "IndoJavE").
- **Sintesis CS dengan LLM**: "Conditioning LLMs to Generate Code-Switched Text" ([arXiv:2502.12924](https://arxiv.org/abs/2502.12924)) — LLM tanpa fine-tuning tidak konsisten menghasilkan CS natural; **COMMIT** (NAACL 2024, [link](https://aclanthology.org/2024.naacl-long.211/)) — instruction tuning code-mixed efektif mengadaptasi LLM English-centric ke bahasa low-resource.
- Dokumen campur EN-ID alami tersedia implisit di CulturaX-id/Reddit Indonesia (bisa di-mining dengan LID per-sentence).

### 4.4 Korpus instruksi/SFT (untuk Jalur B)

| Dataset | Isi | Sumber |
|---|---|---|
| **Aya Collection** | 513 juta instance, >100 bahasa termasuk **jv & su** | [HF CohereForAI](https://huggingface.co/datasets/CohereForAI/aya_collection) |
| **Cendol Collection** | instruksi 23 tugas × 10 bahasa (Indonesia + daerah) | [GitHub IndoNLP/cendol](https://github.com/IndoNLP/cendol) |
| **Bactrian-X id** | ~67k pasangan instruksi id (terjemahan mesin) | [GitHub mbzuai-nlp](https://github.com/mbzuai-nlp/bactrian-x) |
| **Sahabat-AI instruct** | ~448k pasangan id | [model card](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-instruct) |
| **OASST1 id** | subset kecil | [HF](https://huggingface.co/datasets/OpenAssistant/oasst1) |

### 4.5 Benchmark evaluasi

- **IndoMMLU** (14.906 soal), **Nusantara/NusaNLU** ([arXiv:2212.09648](https://arxiv.org/abs/2212.09648)), **SEA HELM/BHASA** ([arXiv:2503.02361](https://arxiv.org/abs/2503.02361), [leaderboard](https://aisingapore.github.io/seahelm-bhasa-leaderboard/)), **IndoMTEB** ([LazarusNLP](https://github.com/LazarusNLP); masuk MMTEB [arXiv:2502.13443](https://arxiv.org/abs/2502.13443)), **IndoCareer** ([arXiv:2409.08564](https://arxiv.org/abs/2409.08564)), leaderboard [indonesian-nlp](https://huggingface.co/spaces/indonesian-nlp/open-indonesian-llm-leaderboard) & [Sahabat-AI](https://huggingface.co/spaces/Sahabat-AI/Sahabat-AI-Leaderboard).
- **Code-switching: tidak ada benchmark CS EN-ID-daerah yang matang** — ini gap nyata; buat eval internal (IndoRobusta-style + sampel Winata/Hidayatullah).

---

## Bagian 5 — Jalur B: Model generatif alternatif (playbook adaptasi bahasa)

> Berlaku jika tujuan akhir Anda adalah model percakapan/generatif id + daerah + en + CS. Laya tidak dipakai sama sekali di jalur ini.

### 5.1 Pilihan basis — cek dulu apakah perlu retrain sama sekali

| Basis | Kenapa | Lisensi |
|---|---|---|
| **Sahabat-AI 8B/9B (instruct)** | **Sudah** continued-pretrain id + jv + su + en oleh GoTo/Indosat — mendukung id, jv, su, bal, batak + en; mungkin cukup di-fine-tune saja untuk CS | [HF GoToCompany](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-instruct) (Llama 3 Community License) |
| **SEA-LION v3.5** | CPT ~200B token data Asia Tenggara di atas Gemma-2 9B / Llama-3.1 8B | [GitHub aisingapore/sealion](https://github.com/aisingapore/sealion), [arXiv:2504.05747](https://arxiv.org/abs/2504.05747) |
| **Qwen3-0.6B/1.7B / Gemma-3-1B / Llama-3.2-1B** | Jika memang butuh kelas kecil ~1B dan mau adaptasi sendiri penuh | Qwen (Apache-2.0 sebagian varian), Gemma Terms, Llama Community License |

### 5.2 Preseden prior art (angka presisi yang bisa ditiru)

| Model | Base | Data | Teknik kunci | Sumber |
|---|---|---|---|---|
| **Sahabat-AI v1** | Llama-3-8B (CPT) | 50B token: id 27,5B + EN/global (Dolma) + jv 0,4B×3,8 + su 0,2B×3,8 | **Tanpa vocab extension** (tokenizer Llama-3 128k cukup); LR 1e-5, batch 256, bf16, 32×H100 ~5 hari; SFT 448k instruksi id | [model card](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) |
| **SEA-LION v3/v3.5** | Gemma-2 9B (CPT ~200B token) | korpus SEALD | analisis token fertility per bahasa SEA sebagai dasar keputusan tokenizer | [arXiv:2504.05747](https://arxiv.org/abs/2504.05747) |
| **Komodo-7B** | Llama-2-7B | hanya 8,5B token id+regional | **vocabulary expansion** + incremental pretraining; mendukung 11 bahasa regional | [HF](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base), [blog](https://tech.yellow.ai/komodo-7b-the-first-llm-for-regional-languages-in-indonesia-2a5256d09ffa) |
| **Swallow (Jepang)** | Llama-2 7–70B | 20–100B token ja + 10% replay en | extend vocab +20% dengan **mean-init embedding** → reduksi token 56%, generasi 78% lebih cepat; gain terbesar di 20B token pertama; EN turun 2–5 poin | [arXiv:2404.17790](https://arxiv.org/abs/2404.17790) |
| **Chinese-LLaMA-Alpaca** | Llama | +20k token zh (32k → 49.953) | mean-init embedding baru; **CPT dua tahap LoRA** (2,97% → 6,06% parameter) | [arXiv:2304.08177](https://arxiv.org/abs/2304.08177), [wiki training](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Training-Details) |
| **Bakpia-V1-0.5B-Javanese** | model kecil open | data jv | instruct-tuned khusus Jawa — bukti kelayakan kelas 0,5B | [HF](https://huggingface.co/afrizalha/Bakpia-V1-0.5B-Javanese) |

*(Catatan: "Nusantara (GoTo)", "Merdeka", "Merah Putih", "Bhasa", "Aquila" — tidak ditemukan dokumentasi teknis/open weights yang kredibel; jangan dijadikan rujukan. Garuda (Indosat-Tech Mahindra, 1,2B param, 16B token) tidak punya open weights yang terverifikasi.)*

### 5.3 Resep end-to-end untuk model ~1B generatif

1. **Tokenizer**: ukur fertility id/jv/su pada base. Jika base Latin-friendly (Qwen 151k / Gemma / Llama-3 128k) → **pertahankan vocab** (preseden Sahabat-AI). Extend hanya jika fertility target > ~1,5–2× bahasa sumber dan ada ≥ beberapa M token; tiap ~10k token baru ≈ +20M parameter pada model 1B yang harus belajar dari nol (preseden extend: Chinese-LLaMA/Swallow/Komodo, selalu mean-init).
2. **CPT full fine-tune bf16**: 10–30B token (gain terbesar di 20B pertama — kurva Swallow). Mix: **55–70% Indonesia** (CulturaX-id + Indo4B + Wiki/News) + **10–30% EN replay** (Dolma/C4) + **5–15% daerah** (jv/su oversample ×3–4; lainnya via NLLB mining + Wikipedia + NusaWrites) + **2–5% dokumen CS EN-ID asli** (mining dari web). Hyperparameter preseden: LR 5e-5–2e-4 (1e-4@7B Swallow; 1e-5@8B Sahabat-AI), warmup ~1k step, cosine decay, global batch ~0,5–1M token, AdamW β=(0,9;0,95), wd 0,1, FlashAttention-2.
3. **SFT**: Aya (id/jav/su) + Cendol + Bactrian-X id + OASST id + **instruksi code-mixed sintetis** (resep [COMMIT](https://aclanthology.org/2024.naacl-long.211/) & [arXiv:2502.12924](https://arxiv.org/abs/2502.12924)) + instruksi EN (jaga kemampuan instruct EN). LoRA/QLoRA biasanya cukup di tahap ini.
4. **Evaluasi**: IndoMMLU + NusaNLU + SEA HELM (id/jv/su) + IndoMTEB + eval EN ringan + **eval CS buatan sendiri** (IndoRobusta-style).
5. **Mitigasi forgetting EN**: replay 10–30% EN; LoRA "learns less, forgets less" ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)); model merging pasca-training ([survey NVIDIA](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)).

### 5.4 Estimasi compute (model ~1B, 20B token, full FT ≈ 1,2×10²⁰ FLOPs; kalibrasi datapoint [TinyLlama](https://arxiv.org/abs/2402.17777))

| Hardware | Estimasi waktu |
|---|---|
| 1× A100 80GB | ~7–11 hari (MFU 40–50%) |
| 8× A100 | ~1–1,5 hari |
| 1× RTX 4090 24GB | ~3–4 minggu |
| SFT/QLoRA (1–5B token) di 1× 24GB | jam–2 hari |

VRAM full FT 1B: bobot bf16 ~2GB + optimizer ~12–16GB + aktivasi (grad ckpt) ~4GB ≈ 18–22GB → **1 GPU konsumen 24GB cukup** (optimizer 8-bit bila sempit). Tools: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (one-stop, GUI), [torchtune](https://github.com/pytorch/torchtune) (native, dukung vocab resize), [Unsloth](https://github.com/unslothai/unsloth) (SFT/LoRA cepat), [axolotl](https://github.com/axolotl-ai-cloud/axolotl), HF transformers+accelerate.

**Scratch vs CPT vs LoRA**: scratch ~1B butuh ~1–3T token (TinyLlama: 3T token, 16×A100, 90 hari) — tidak ekonomis; CPT full = default terbukti; LoRA-CPT = opsi hemat dengan ceiling lebih rendah untuk bahasa yang sangat berbeda dari base.

---

## Bagian 6 — Risiko, lisensi, & hal yang tidak ditemukan

| Risiko | Fakta & mitigasi |
|---|---|
| **Salah paham fundamental** | Laya tidak generatif — pastikan use case sebelum investasi. |
| **Lisensi** | Laya: Apache-2.0 (aman, komersial boleh). Dataset: **NusaX CC-BY-SA 4.0 (share-alike!)**, CulturaX/mC4 terikat kebijakan Common Crawl. Base generatif: Llama 3 Community License (>700M MAU perlu izin), Gemma Terms, Qwen sebagian Apache-2.0. |
| **Degradasi EN** | Terdokumentasi 2–5 poin dengan replay 10% (Swallow); replay 10–30% + LoRA/merging. |
| **Data daerah sangat kecil** | jv/su 0,2–0,4B token unik saja; lainnya ~1k kalimat → oversample ×3–4, NLLB back-translation, NusaWrites curation, sintesis LLM. |
| **Benchmark CS belum matang** | Bangun eval CS internal + human eval. |
| **Fertility tokenizer tak terdokumentasi** (khusus Laya) | Ukur sendiri sebelum memutuskan apa pun tentang vocab. |
| **"Model hantu"** | Garuda/Merdeka/Merah Putih/Bhasa/Nusantara-GoTo tidak terdokumentasi — jangan jadikan rujukan teknis. |

**Tidak ditemukan / tidak terverifikasi** (jujur dari riset): provenance mmBERT; korpus training Laya; angka token CulturaX-id; ID arXiv IndoMTEB & CodeMixBench (judulnya ada, tautan stabil tidak terverifikasi); dataset "SANME/SanmeTalks/IndoVoA" (kemungkinan nama keliru); "SALSA"/"UMBC" sebagai dataset CS EN-ID (yang ada: IndoCollex & kamus alay — soal bahasa gaul, bukan CS).

---

## Bagian 7 — Langkah berikutnya yang disarankan

1. **Putuskan use case**: lapisan keputusan/klasifikasi (→ Jalur A) atau percakapan/generatif (→ Jalur B). Ini keputusan yang menentukan segalanya.
2. **Eksperimen murah Jalur A (1–2 hari)**: pull `laya-multilingual` → patch router → evaluasi baseline pada MASSIVE-id + NusaX-senti → fine-tune dengan notebook Kaggle pada MASSIVE-id → ukur delta.
3. **Eksperimen murah Jalur B**: coba dulu Sahabat-AI/SEA-LION instruct dengan prompt CS — mungkin 80% kebutuhan tercapai tanpa training apapun; baru putuskan CPT.
4. **Ukur token fertility** id/jv/su/en pada kandidat tokenizer apa pun sebelum training (metodologi SEA-LION).
5. Kumpulkan & konversi data ke format target (typed-decisions untuk A; CPT/SFT mix untuk B) — lihat Bagian 3.2 & 4.

---

## Lampiran — Semua sumber primer

**Laya**: [HF laya](https://huggingface.co/convaiinnovations/laya) · [HF laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) · [HF laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) · [GitHub NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) · [notebook fine-tune Kaggle](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) · [hasil sweep 51 bahasa](https://raw.githubusercontent.com/NandhaKishorM/laya/research/research/results/cpu_51_language_sweep.json) · [PR #286 router](https://github.com/NandhaKishorM/laya/pull/286) · [issue #320](https://github.com/NandhaKishorM/laya/issues/320) · [laya.convaiinnovations.com](https://laya.convaiinnovations.com)

**Prior art & teknik**: [Sahabat-AI](https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base) · [SEA-LION](https://arxiv.org/abs/2504.05747) · [Swallow](https://arxiv.org/abs/2404.17790) · [Chinese-LLaMA-Alpaca](https://arxiv.org/abs/2304.08177) · [Komodo-7B](https://huggingface.co/Yellow-AI-NLP/komodo-7b-base) · [LoRA learns less and forgets less](https://arxiv.org/abs/2405.09673) · [QLoRA](https://arxiv.org/abs/2305.14314) · [TinyLlama](https://arxiv.org/abs/2402.17777) · [CulturaX](https://arxiv.org/abs/2309.14050) · [NusaBERT](https://arxiv.org/abs/2403.01652) · [NusaMT-7B](https://arxiv.org/abs/2410.10461)

**Data Indonesia/daerah/CS**: [Indo4B](https://huggingface.co/datasets/SEACrowd/indo4b) · [NusaX](https://github.com/IndoNLP/nusax) · [NusaCrowd](https://github.com/IndoNLP/nusa-crowd) · [NusaWrites](https://github.com/IndoNLP/nusa-writes) · [Cendol](https://github.com/IndoNLP/cendol) · [NLLB](https://huggingface.co/datasets/allenai/nllb) · [Aya Collection](https://huggingface.co/datasets/CohereForAI/aya_collection) · [Bactrian-X](https://github.com/mbzuai-nlp/bactrian-x) · [IndoRobusta](https://arxiv.org/abs/2211.05360) · [COMMIT](https://aclanthology.org/2024.naacl-long.211/) · [Conditioning LLMs for CS](https://arxiv.org/abs/2502.12924) · [IndoMMLU](https://huggingface.co/datasets/indolem/IndoMMLU) · [SEA HELM](https://arxiv.org/abs/2503.02361) · [IndoCollex/normalisasi](https://github.com/NormalisasiKata/NormalisasiKata) · [awesome-code-mixing](https://github.com/lingo-iitgn/awesome-code-mixing)
