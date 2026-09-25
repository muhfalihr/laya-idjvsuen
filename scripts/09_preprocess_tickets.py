"""Preprocessing dataset ticket_messages menjadi test set bersih untuk laya-idjvsuen.

Tahapan:
1. Normalisasi teks: whitespace, karakter kontrol, tanda kutip melengkung.
2. Masking PII & noise: URL, email, nomor telepon, hex/angka panjang -> <URL> <EMAIL> <PHONE> <ID>.
3. Buang: pesan kosong/terlalu pendek, perintah bot (/start, /ticket ...), pesan murni URL/angka,
   duplikat persis (hash), duplikat near-dup (teks ternormalisasi sama).
4. Statistik: per sumber, per message_from, distribusi panjang, kandidat code-switching
   (mengandung kata fungsi id DAN en), pesan panjang >=4 kata.
5. Output terpisah user / handler (untuk evaluasi model, pesan user biasanya objek klasifikasi;
   pesan handler berguna sebagai data pendukung).

Output:
  data/processed/tickets_user.jsonl      (pesan dari user, bersih)
  data/processed/tickets_handler.jsonl   (pesan dari handler, bersih)
  data/processed/tickets_stats.json      (laporan preprocessing)
"""
import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "tickets", "ticket_messages_raw.jsonl")
PROC = os.path.join(ROOT, "data", "processed")

RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_PHONE = re.compile(r"(?<!\w)(\+62|0)8\d{7,12}(?!\w)|(?<!\w)\+?\d[\d\s-]{9,14}\d(?!\w)")
RE_HEXID = re.compile(r"\b[0-9a-f]{12,40}\b", re.I)
# Secret / API key: token vendor umum + pola key=value yang bernama sensitif
RE_SECRET_TOKEN = re.compile(
    r"\b(AIza[0-9A-Za-z_\-*]{20,}|sk-[A-Za-z0-9_\-*]{10,}|gh[pousr]_[A-Za-z0-9*]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,})")
RE_KV_SECRET = re.compile(
    r"(?i)\b((?:api[_-]?key|pass(word)?|passwd|pwd|secret|token|credential|private[_-]?key)"
    r"\s*[=:]\s*)([^\s,;'\"]+)")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
RE_WS = re.compile(r"\s+")
RE_BOTCMD = re.compile(r"^/\w+(\s|$)")
RE_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\U00002190-\U000021FF]+")

# Kata fungsi penanda bahasa (heuristic ringan, cukup untuk statistik mix)
ID_WORDS = {"yang", "dan", "di", "ke", "dari", "gak", "ga", "nggak", "ngga", "udah", "sudah",
            "belum", "iya", "apa", "kapan", "kok", "aja", "saja", "mas", "mbak", "pak", "bu",
            "kak", "banget", "bgt", "kenapa", "gimana", "bagaimana", "tolong", "makasih",
            "terima", "kasih", "maaf", "sama", "ini", "itu", "ada", "bisa", " tidak", "ngomong"}
EN_WORDS = {"the", "and", "is", "are", "was", "you", "your", "please", "thanks", "thank",
            "sorry", "ok", "okay", "yes", "no", "not", "can", "will", "have", "has", "for",
            "with", "this", "that", "what", "when", "why", "how", "just", "wait", "done",
            "error", "problem", "issue", "help", "me", "my", "i"}


def normalize(text, drop_emoji=True):
    text = unicodedata.normalize("NFKC", text or "")
    text = RE_CTRL.sub(" ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = RE_URL.sub(" <URL> ", text)
    text = RE_EMAIL.sub(" <EMAIL> ", text)
    text = RE_PHONE.sub(" <PHONE> ", text)
    text = RE_HEXID.sub(" <ID> ", text)
    text = RE_SECRET_TOKEN.sub(" <SECRET> ", text)
    text = RE_KV_SECRET.sub(lambda m: m.group(1) + "<SECRET>", text)
    if drop_emoji:
        text = RE_EMOJI.sub(" ", text)
    text = RE_WS.sub(" ", text).strip()
    return text


def keep(text):
    """Filter kelayakan sebuah pesan sebagai sampel test."""
    if len(text) < 4:
        return False
    if RE_BOTCMD.match(text):
        return False
    words = text.split()
    if len(words) < 2:
        return False
    # murni placeholder/angka/tanda baca tanpa kata bermakna
    real = [w for w in words if re.search(r"[a-zA-Z]", w)]
    if len(real) < 2:
        return False
    return True


def lang_mix(text):
    """('id' | 'en' | 'mixed' | 'other') berdasar kata fungsi."""
    toks = set(re.findall(r"[a-zA-Z']+", text.lower()))
    n_id, n_en = len(toks & ID_WORDS), len(toks & EN_WORDS)
    if n_id and n_en:
        return "mixed"
    if n_id:
        return "id"
    if n_en and len(toks) >= 3:
        return "en"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=RAW)
    p.add_argument("--min-words", type=int, default=3,
                   help="panjang minimum (kata) pesan untuk dataset final")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.raw, encoding="utf-8")]
    stats = {"raw_total": len(rows), "dropped": Counter(), "by_source_kept": Counter(),
             "by_from_kept": Counter(), "lang": Counter(), "cs_candidates": 0}

    seen_exact, seen_norm = set(), set()
    out_user, out_handler = [], []
    for r in rows:
        text = normalize(r["message"])
        if not keep(text):
            stats["dropped"]["filtered"] += 1
            continue
        h = hashlib.md5((r["source"] + text).encode()).hexdigest()
        if h in seen_exact:
            stats["dropped"]["dup_exact"] += 1
            continue
        seen_exact.add(h)
        # near-dup: lowercase tanpa placeholder
        norm_key = re.sub(r"\s+", " ", re.sub(r"<\w+>", "", text.lower())).strip()
        if norm_key and norm_key in seen_norm:
            stats["dropped"]["dup_near"] += 1
            continue
        seen_norm.add(norm_key)

        if len(text.split()) < args.min_words:
            stats["dropped"]["too_short"] += 1
            continue

        mix = lang_mix(text)
        stats["lang"][mix] += 1
        if mix == "mixed":
            stats["cs_candidates"] += 1
        stats["by_source_kept"][r["source"]] += 1
        stats["by_from_kept"][r["message_from"] or "?"] += 1

        item = {"text": text, "source": r["source"], "ticket_id": r["ticket_id"],
                "message_from": r["message_from"], "lang_guess": mix,
                "ts": r["timestamp"][:19]}
        (out_user if r["message_from"] == "user" else out_handler).append(item)

    os.makedirs(PROC, exist_ok=True)
    for name, items in (("tickets_user.jsonl", out_user), ("tickets_handler.jsonl", out_handler)):
        with open(os.path.join(PROC, name), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    lens = sorted(len(it["text"].split()) for it in out_user + out_handler)
    n = len(lens)
    stats["kept_total"] = n
    stats["kept_user"], stats["kept_handler"] = len(out_user), len(out_handler)
    stats["len_words"] = {"p50": lens[n // 2] if n else 0,
                          "p90": lens[int(n * 0.9)] if n else 0,
                          "p99": lens[int(n * 0.99)] if n else 0,
                          "max": lens[-1] if n else 0}
    stats["dropped"] = dict(stats["dropped"])
    stats["lang"] = dict(stats["lang"])
    stats["by_source_kept"] = dict(stats["by_source_kept"])
    stats["by_from_kept"] = dict(stats["by_from_kept"])
    with open(os.path.join(PROC, "tickets_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\noutput: tickets_user.jsonl ({len(out_user)}), tickets_handler.jsonl ({len(out_handler)}) di {PROC}")


if __name__ == "__main__":
    main()
