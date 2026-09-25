# Benchmark: Laya (tanpa fine-tune) vs Jev (OpenRouter) vs laya-idjvsuen

**Bahasa / Language**: **Indonesia** | [English](BENCHMARK.en.md)

Uji: klasifikasi **12 kategori domain tiket internal** pada pesan support nyata
(bahasa Indonesia santai + campuran). Semua peserta dievaluasi pada **300 sampel
identik**, opsi & instruksi identik, gold = weak labels (aturan kata kunci — bukan
anotasi manusia).

## Hasil utama (300 sampel identik)

| Model | Akurasi | Macro-F1 | ECE | Latensi | Biaya |
|---|---|---|---|---|---|
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) (tanpa fine-tune) | 24,7% | 0,313 | 0,536 | lokal, ~ms | $0 |
| [`laya-idjvsuen-v1`](https://huggingface.co/faall7479/laya-idjvsuen-v1) (id/jv/su/en umum) | 44,7% | 0,354 | 0,117 | lokal, ~ms | $0 |
| Jev 1.13 (`~typesafe/jev-latest`, OpenRouter) | 84,7% | 0,564 | 0,083 | 0,32 s (API) | $0,0093 |
| `laya-idjvsuen-v2` (fine-tune domain; digantikan v3, tidak dipublish) | **96,0%** | **0,752** | 0,030 | lokal, ~ms | $0 |
| [`laya-idjvsuen-v3`](https://huggingface.co/faall7479/laya-idjvsuen-v3) (domain + kelas langka) | **93,3%** | 0,680 | **0,015** | lokal, ~ms | $0 |

v2 unggul 2,7 poin akurasi keseluruhan tetapi lemah di kelas langka; v3 dipilih sebagai
rilisan karena seimbang per-kelas (macro per-kelas 0,830 → 0,886, ECE terbaik) — lihat
detail di laporan v3.

Test penuh 1.126 item: v2 = 95,2% akurasi / 0,787 macro-F1 / ECE 0,024.

## Regresi anti-lupa (v2 vs v1, 15 test set lama)

Replay 12.000 item lama saat training v2 menjaga kemampuan sebelumnya: intent
id/en/jv/su turun ≤1,6 poin, sentimen Sunda malah naik +3,8 poin, code-switch
stabil (±1 poin). Tanpa replay, degradasi khas jauh lebih besar.

## Metodologi

- **Peserta laya**: forward path SDK resmi (temperatur dari config checkpoint), tanpa router.
- **Jev**: endpoint OpenRouter `/api/alpha/decisions` (skema kompatibel laya:
  state + questions{type, instructions, criteria}); `~typesafe/jev-latest`
  ter-resolve ke `typesafe/jev-1.13-20260917`; temperature API default; 0 kegagalan API.
- Skrip: `scripts/13_bench_openrouter.py` (Jev) dan `scripts/05_eval.py` (laya);
  file hasil `data/processed/eval_{base_ticket300,v1_ticket300,jev,v2_ticket300}.json`.

## Catatan jujur

1. **Gold = weak labels.** Keunggulan v2 sebagian adalah "belajar distribusi
   annotator (aturan)". Angka 84,7% Jev secara zero-shot terhadap label yang sama
   tetap sangat kuat untuk model generik. Untuk angka netral-manusia, perlu audit
   manual sampel (±200–500 pesan).
2. **Kelas langka** (security, compliance, platforms) nyaris tanpa data training —
   akurasi kelas ini rendah untuk semua peserta; macro-F1 memperlihatkan gap ini.
3. Jev adalah model keputusan generik ber-hosting — benchmark ini menunjukkan nilai
   fine-tuning domain, bukan kelemahan Jev sebagai produk.
4. Biaya/latensi: laya berjalan lokal (CPU pun cukup, 322M); Jev butuh round-trip API.
