# Riset Mendalam: Jev (TypeSafe AI) & Analisis Gap vs laya-idjvsuen

**Bahasa**: **Indonesia** | [English](RISET-JEV.en.md)

> Disusun 26 September 2026. Semua klaim bersumber primer (docs.typesafe.ai, blog TypeSafe,
> evals.typesafe.ai, OpenRouter, HuggingFace API, legal/mca) kecuali yang ditandai eksplisit.
> Dilengkapi analisis 8 screenshot thread penjelas @techwith.ram yang dilampirkan.

---

## Ringkasan Eksekutif (baca ini dulu)

1. **Jev adalah satu kelas model dengan Laya** — *System One / decision model* non-generatif:
   input `state` (teks tak terstruktur) + pertanyaan terketik (`choice`/`score`/`noul`),
   output keputusan + probabilitas terkalibrasi, tanpa generasi teks. Format I/O-nya nyaris
   identik dengan format laya (`build_sequence`). Perbedaannya bukan *jenis* model, melainkan
   **generalitas, skala, dan ekosistem**.
2. **Jev fully proprietary** — ukuran parameter tidak pernah diungkap, tidak ada weights,
   tidak ada org HuggingFace resmi, tidak ada fine-tuning untuk customer. Hanya API
   (`POST /v1/systemone`, di OpenRouter via `~typesafe/jev-latest`).
3. **Metode training-nya = RLCD** ("Reinforcement Learning for Calibrated Decisions") —
   jalur ketiga setelah RLHF/RLVR. Korpus, ukuran data, dan loss tidak dipublikasikan.
   Pipeline laya-idjvsuen (RLCD REINFORCE + proper scoring rules + soft CE) secara konsep
   adalah resep yang sama.
4. **Jev bukan yang terpintar di tugas domain sempit**: evals resmi TypeSafe sendiri
   menempatkan Jev 67,8% overall — **kalah dari model frontier** (sol 74,1%) dan hanya 76,0%
   di Customer Service (vs sol 78,3%). Keunggulan Jev adalah **harga/latensi** (orde magnitude
   lebih murah & cepat) dan generalitas lintas domain. Ini konsisten dengan benchmark kami:
   laya-idjvsuen-v3 93,3% vs Jev 84,7% pada 12-kategori routing tiket internal.
5. **Jev mengakui lemah di bahasa non-Inggris** — moat alami laya-idjvsuen (id/jv/su/CS).
6. **⚠️ Legal**: ToS TypeSafe (MCA §2.3(b)) **melarang distilasi dari output Jev**. Jangan
   melatih model apa pun dengan output Jev. Benchmark & publikasi hasil evaluasi diperbolehkan
   (tidak ada klausul yang melarang).
7. **Jawaban singkat pertanyaan "bisakah laya-idjvsuen sepintar Jev?"**: *pada domain yang
   dilatih — sudah terjadi dan bisa diperlebar; sebagai model keputusan umum lintas domain —
   tidak dengan backbone 322M, butuh base 1–9B+; menyalin Jev dari output-nya — dilarang ToS.*
   Roadmap lengkap: [Bagian 8](#bagian-8--bisakah-laya-idjvsuen-menjadi-sepintar-jev).

---

## Bagian 1 — Apa itu Jev

### 1.1 Identitas

| Aspek | Fakta (sumber primer) |
|---|---|
| Pengembang | TypeSafe AI — CEO **Diogo Almeida** (co-inventor RLHF/InstructGPT, ex-Google Brain), CTO Erik Gafni, COO Sasha Sheng. Keluar stealth 15 Sep 2026 (pendanaan $40M — sumber sekunder, belum terverifikasi primer) |
| Versi | Satu-satunya rilis resmi: **`jev-1.13.0`** (alias `jev-latest`; `jev-preview` juga menunjuk ke sana) |
| Parameter | **Tidak diungkapkan** (blog menyebut "a new model architecture, parallel sampler" tanpa detail) |
| Akses | Hanya API hosted: `POST /v1/systemone`; OpenRouter `~typesafe/jev-latest`, `typesafe/jev-router` |
| Weights | **Tidak ada** — org HF `typesafe`/`typesafeai`/`typesafe-ai` semuanya kosong |
| Fine-tune customer | **Tidak tersedia** |

### 1.2 Spesifikasi teknis vs laya

| Kemampuan | Jev 1.13 | laya / laya-idjvsuen |
|---|---|---|
| Primitif | Choice (maks **255** opsi), Score (rubric berjenjang), Noul | Choice, Score, Noul — sama |
| Pertanyaan per call | Banyak, paralel ("adding questions barely changes response time") | Banyak (loop per qid; sekali forward per sekuens) |
| Context | **64k token** (32k state + pertanyaan) | 768–1024 token (config run kami 768) |
| Criteria terstruktur | Ya — options/levels/instructions bisa JSON | Ya — `criteria` dict/list |
| Modalitas | Teks saja | Teks saja |
| Output | `choice`/`score` + `probabilities` + `confidence` | sama + kalibrasi suhu per-tipe |
| Latency klaim | 70–500 ms | lokal, tergantung hardware (~ratusan ms di GPU konsumen) |
| Self-host | Tidak | **Ya** — Apache-2.0 / CC-BY-SA-4.0 |

### 1.3 "Open-Jev" 2B/9B/27B bukan produk TypeSafe

Proyek reproduksi independen oleh Zefan Cai (Qwen3.5/3.8 base + decision head, LoRA rank 8
untuk 27B). Kode MIT, adapter Apache-2.0, dataset publik `Open-Jev` / `Open-Jev-v1.1`
(**326.619–408.884 baris** keputusan terketik dari 25 sumber task) — ini legal untuk
dipakai training (lihat Bagian 8). JevBench pihak ketiga: Jev 1.13 = 86,6% overall /
73,0% Hard; Open-Jev-27B-v1.1 = 85,3% / 72,1% Hard.

---

## Bagian 2 — Kemampuan & benchmark

### 2.1 Evals resmi TypeSafe (evals.typesafe.ai; referensi = rata-rata GPT-6 Astra & Claude Fable 5.1)

| Model | Overall | Security | Agent Trace | Invoice | Customer Service |
|---|---|---|---|---|---|
| **Jev** | **67,8%** · $0,0004 · 0,4s | 61,7% | 71,6% | 61,8% | 76,0% · $0,0001 |
| sol (GPT) | **74,1%** | 62,5% | **76,6%** | **79,1%** | **78,3%** |
| opus 5 | 73,1% | **66,2%** | 75,2% | 78,4% | 72,4% |
| terra (GPT) | 67,9% | 51,2% | 73,0% | 74,7% | 72,7% |
| sonnet 5 | 67,8% | 60,8% | 68,0% | 72,9% | 69,3% |
| DS v4 flash | 64,4% | 37,9% | 73,0% | — | 76,8% |
| haiku 4.5 | 53,6% | 58,8% | 57,2% | 42,9% | 55,4% |

Pola penting: **Jev menang di biaya/latensi (klaim 193,6× lebih cepat, 444,6× lebih murah),
kalah di akurasi rata-rata dari model frontier.** Homepage: $0,000081/workflow (0,114s) vs
$0,013880 (8,566s) untuk LLM. Harga resmi: **$42/miliar token input, output gratis**;
rate limit 250k token/s, 1.200 req/menit.

### 2.2 Jaggedness — 9 kelemahan yang diakui TypeSafe sendiri (docs.model-jaggedness)

1. **Literal reading** — menjawab pertanyaan yang tertulis, bukan yang dimaksud (negasi/scoping harfiah).
2. **Bukan kalkulator** — counting tak andal, hex/RGB lemah, kalibrasi numerik buruk.
3. **Tanggal = teks** — tidak dibaca sebagai besaran terurut; quarter/window lemah.
4. **Indirection** — instruksi double-negative kurang andal.
5. **Context rot** — akurasi turun saat state berisi konten tak relevan.
6. **Tidak adversarial by default** — instruksi ter-inject "can move the answer".
7. **Kontradiksi instruksi/criteria** menurunkan performa.
8. **Invarian struktural tidak dijamin** — contoh nyata: Noul refund 0,22 vs Choice no 0,99
   (confidence 0,97); pasangan refund/not_refund 0,72/0,47 (jumlah 1,19 — melanggar P/1−P).
9. **Chained choices untuk generate text** "will not work well and will be very slow".

Plus dari models.md: **bahasa non-Inggris lebih lemah** (CJK disebut eksplisit).

### 2.3 Kredibilitas klaim

TypeSafe terbuka soal biasnya sendiri: eval dari laptop di West Coast, label referensi =
produk GPT/Claide, workflow gain "on the higher end", klaim 0% hallucination "is not
empirical" (dijamin schema, bukan diukur). Tidak ada perbandingan vs Laya di sumber primer
mana pun. Ada juga koreksi sirkulasi publik: founder TypeSafe = Diogo Almeida, **bukan**
Joseph Perla (dia founder TrustedRouter, reseller akses Jev).

### 2.4 Dari screenshot yang dilampirkan (thread @techwith.ram)

- **Slide 07**: "cannot hallucinate" perlu konteks — schema error (label invalid) mustahil,
  tapi **decision error** (label valid tapi salah) tetap bisa. *Valid ≠ correct.*
- **Slide 08**: Jev tidak menggantikan LLM — **membungkusnya** (pilih model → LLM bekerja →
  setujui tool call → verifikasi hasil). Jev handle keputusan, LLM handle penalaran.
- **Slide 09**: sweet spot Jev = keputusan **semantik + terbatas + berulang** (routing,
  triage, rerank, guardrail, labeling volume tinggi). Terbuka → LLM; aturan deterministik → kode.
- **Slide 06**: nilai jual utama = **probabilitas + confidence**, bukan hanya jawaban —
  threshold untuk automasi (sama dengan nilai jual laya).

---

## Bagian 3 — Data & training

- Metode resmi: **RLCD** — diposisikan sebagai jalan ketiga setelah RLHF (sycophancy,
  halusinasi percaya-diri) dan RLVR (lambat/mahal). Kontrak output: model tidak generasi teks,
  hanya keputusan + probabilitas; kalibrasi group-level (0,2 → terjadi ~20% waktu).
- **Tidak dipublikasikan**: korpus, ukuran data, loss function detail, SFT pipeline,
  distilasi. Blog punya heading FAQ "What is Jev trained on?" — isinya kosong. Tidak ada paper/arXiv.
- Tidak melatih di data customer; ZDR enterprise tersedia.

---

## Bagian 4 — Ekosistem

SDK Python (v0.5.7 → v0.7.1 dalam sepekan) + SDK JavaScript (~30 hal.), agent skill untuk
Claude Code/Codex, adapter GitHub, playground console.typesafe.ai, situs evals publik,
~20 cookbook (re-ranking CLERC 5%→18% top-1, guardrails, klasifikasi hierarkis 75 industri).

---

## Bagian 5 — Kepatuhan & legal (PENTING)

- **Benchmark diperbolehkan** — tidak ada klausul ToS yang melarang evaluasi/publikasi hasil.
- **§2.3(b) Master Customer Agreement MELARANG**: memakai Services/Output untuk
  *"model distillation, train a model to imitate the output, or develop a similar or competing
  product or service"*. Konsekuensi praktis:
  1. **Jangan pernah melatih laya-idjvsuen (atau model apa pun) dengan output Jev.**
  2. Benchmark API (skrip 13) aman — itu evaluasi, bukan training.
  3. Klausul "competing product" berpotensi luas jika menjadi customer TypeSafe —
     konsultasi hukum jika hendak mengkomersialkan.

---

## Bagian 6 — Posisi laya-idjvsuen hari ini (data kami)

| | laya-multilingual (base) | **laya-idjvsuen-v3** | Jev 1.13 |
|---|---|---|---|
| 12-kategori tiket internal (300 sampel identik) | 24,7% | **93,3%** (ECE 0,015) | 84,7% (ECE 0,083) |
| Biaya per 1k keputusan | gratis (lokal) | gratis (lokal, GPU 8GB) | ~$0,0093 di benchmark kami ($0,042/MTok input) |
| Latency rata-rata (benchmark kami) | lokal | lokal | 0,32s |
| Bahasa id/jv/su/CS | lemah | **kuat** | lemah (diakui sendiri) |
| Generalitas lintas domain | — | sedang (MASSIVE 60-kelas + domain) | **kuat** |
| Self-host / privacy data internal | ya | **ya** | tidak |

Kemenangan v3 atas Jev di domain sempit **konsisten** dengan evals resmi TypeSafe (Jev
kalah dari model frontier di Invoice/Customer Service) — bukan anomali.

---

## Bagian 7 — Apa yang membuat Jev "pintar"

Tiga sumber keunggulan Jev, dan mana yang bisa direplikasi:

1. **Generalitas (bisa direplikasi sebagian)** — dilatih pada korpus keputusan lintas domain
   yang besar. Setara open-nya: dataset `Open-Jev` (400k+ baris, 25 sumber task, lisensi
   permissif) + MASSIVE + NusaX + data domain sendiri.
2. **Skala backbone (tidak bisa direplikasi di 322M)** — parameter Jef tidak diungkap, tapi
   kemampuan reasoning lintas domainnya menunjukkan base yang jauh lebih besar dan/atau
   pretraining masif. Open-Jev butuh 9–27B untuk mendekati Jev di JevBench Hard
   (2B: 46/111, 9B: 66/111, 27B: 80/111 vs Jev 81/111).
3. **Serving engineering (bisa direplikasi)** — latensi 70–500ms & harga ekstrem murah itu
   hasil optimasi inference, bukan kecerdasan.

---

## Bagian 8 — Bisakah laya-idjvsuen menjadi "sepintar Jev"?

Jawaban jujur, tiga tingkat:

### (1) Bisa, via data/training saja — bahkan sebagian sudah terjadi

- **Domain id/jv/su/en + CS**: sudah menang dari Jev (93,3% vs 84,7%). Resep: data supervised
  domain + RLCD proper-scoring-rules — esensinya sama dengan yang diklaim TypeSafe.
- **Perlebar domain dengan aman & legal**: gabungkan dataset publik `Open-Jev`
  (326–408k keputusan terketik, MIT/Apache) ke pipeline v4 bersama replay data eksisting →
  generalitas naik tanpa menyentuh output Jev.
- **Lampui kelemahan Jev yang diakui sendiri, in-domain**: augmentasi negasi (vs literal
  reading), data tanggal/angka format campuran, criteria kontradiktif, **robustness
  prompt-injection** (latih dengan instruksi tersisip — Jev gagal di sini), state panjang +
  distractor (vs context rot).
- **Kalibrasi**: proper scoring rules kami lebih eksplisit daripada yang didokumentasikan
  TypeSafe; perbaikan datang dari coverage data + hard negatives, bukan arsitektur.
- **Audit set manusia** untuk 12 kategori (TODO lama) → klaim makin keras, bukan "learned
  the annotator".

### (2) Butuh perubahan skala/arsitektur — mungkin, tapi bukan "fine-tune lagi"

- **Generalitas setara Jev**: backbone 322M tidak akan mengejar world-knowledge. Jalur
  realistis: pindah/ganti base ke encoder/decoder 1–9B (jelaskan pola Open-Jev: LoRA +
  decision head di atas Qwen) — tapi hardware 8GB kami hanya cukup untuk 1–2B (LoRA,
  quantized), dan itu proyek beda kelas, bukan "update v4".
- **Context 64k**: perlu long-context encoder atau chunking+aggregation (bisa disimulasikan
  di level pipeline: ringkas state dulu → decide).
- **Multi-pertanyaan paralel per forward** & cardinality 255: butuh modifikasi head —
  sederhana secara konsep (head_max_len 512 kami sudah menampung 60 opsi), tapi bukan gratis.
- **Latensi sub-detik konsisten**: quantization INT8/ONNX + batching di serving.

### (3) Tidak realistis / tidak boleh

- **"Sepintar Jev secara umum" pada 322M** — tidak terjadi lewat fine-tune seberapa pun;
  itu batas kapasitas, bukan data.
- **Menyalin Jev dari output-nya** — **dilarang ToS TypeSafe §2.3(b)**. Jangan.
- **Menandingi ekosistemnya** (SDK dua bahasa, evals publik, cookbook, integrasi) — kerja
  produk bertahun-tahun, bukan kemampuan model.

### Rekomendasi prioritas (konkret)

1. **v4 = generalitas**: mixing `Open-Jev` dataset (publik, legal) + replay MASSIVE/NusaX +
   data tiket → naikkan kemampuan lintas domain tanpa kehilangan domain.
2. **Hardening**: augmentasi negasi + prompt-injection + tanggal/angka → langsung menyerang
   4 dari 9 jaggedness Jev.
3. **Audit manusia** 200–300 tiket → ganti weak labels di test set → angka benchmark jadi
   defensible untuk publikasi.
4. **Serving**: export ONNX/quantized untuk latensi & CPU inference.
5. (Opsional, jangka panjang) **backbone 1–2B** LoRA + decision head di GPU 8GB jika memang
   ingin mengejar generalitas — dengan ekspektasi jujur bahwa itu kelas proyek baru.

---

## Sumber primer

- Blog: <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- Docs model: <https://docs.typesafe.ai/models.md> · primitives: <https://docs.typesafe.ai/primitives.md>
- Jaggedness: <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>
- Evals: <https://evals.typesafe.ai/> · Legal/MCA: <https://typesafe.ai/legal/mca>
- OpenRouter: <https://openrouter.ai/~typesafe/jev-latest>
- Open-Jev (komunitas, bukan TypeSafe): <https://zefan-cai.github.io/open-jev> ·
  [ZefanCai/Open-Jev-9B](https://huggingface.co/ZefanCai/Open-Jev-9B) ·
  [github.com/Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)
- Team: <https://typesafe.ai/team>

*Tidak ditemukan di sumber primer: ukuran parameter Jev, arsitektur detail, korpus/ukuran
data training, isi FAQ blog (heading ada, isi kosong), paper, perbandingan resmi vs Laya,
pricing per-token di halaman OpenRouter (array endpoints kosong saat dicek).*
