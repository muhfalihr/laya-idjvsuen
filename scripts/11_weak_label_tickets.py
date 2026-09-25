"""Weak labeling pesan ticket user ke 12 kategori domain (pilihan user).

Pendekatan: aturan kata kunci (id + en) dengan skor per kategori; label = argmax
bila skor >= threshold dan margin atas peringkat-2 cukup. Pesan ambigu dibiarkan
tanpa label (tidak dipakai training). Ini label lemah (noisy) - evaluasi test
adalah "weak-label accuracy", bukan ground truth manusia.

Output: data/processed/tickets_labeled.jsonl {text, label, score, ticket_id, ...}
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

# Deskripsi ikut menjadi teks opsi di pertanyaan laya: "nama: deskripsi".
LABELS = {
    "asset_devices": "issues or requests about physical assets and devices: laptops, PCs, phones, printers, monitors, peripherals, chargers, device inventory",
    "infrastructure": "servers, VMs, Kubernetes, deployment, environments, storage, backups, databases, hosting infrastructure",
    "platforms": "internal or third-party platforms and systems: portals, dashboards, web apps, SaaS, cloud consoles, specific named systems",
    "employee_support": "general employee and office support: onboarding/offboarding, facilities, office supplies, documents, administrative help",
    "security": "security incidents and concerns: phishing, malware, suspicious activity, vulnerabilities, unauthorized access, security tools",
    "compliance": "compliance, audit, certification, policy, and regulatory requests",
    "account_identity": "accounts and identity: logins, passwords, SSO, 2FA, email accounts, existing-account permission changes, profiles",
    "data_reporting": "data, reports, datasets, exports, analytics, dashboards content, statistics",
    "networks_connectivity": "network and connectivity: internet, VPN, Wi-Fi, LAN, proxy, DNS, remote connectivity",
    "application_issue": "errors, bugs, crashes, or unexpected behavior in an application or service",
    "service": "ticket-flow and general service desk matters: greetings, status updates, follow-ups, confirmations, thanks, closing",
    "request_access": "requests for NEW access, credentials, or permissions to systems, repos, servers, tools, groups, or distribution lists",
}

# Kata kunci (case-insensitive). Bobot default 1; frasa lebih spesifik bobot 2.
KEYWORDS = {
    "asset_devices": ["laptop", "notebook", "printer", "monitor", "keyboard", "mouse",
                      "charger", "adaptor", "handphone", " hp ", "hp gsm", "android", "iphone",
                      "device", "scanner", "ups ", "cctv", "barcode", "sim card"],
    "infrastructure": ["server", "vm ", "vps", "kube", "kubernetes", "deploy", "pod",
                       "docker", "nginx", "haproxy", "backup", "restore", "database",
                       "postgres", "mysql", "mongo", "redis", "ssl", "sertifikat ssl",
                       "domain", "dns record", "load balancer", "vmware", "esxi"],
    "platforms": ["platform", "portal", "dashboard", "console", "web app", "sistem",
                  "aplikasi web", "gitlab", "jira", "confluence", "sonar", "argus",
                  "digibox", "grafana", "n8n", "metabase", "superset"],
    "employee_support": ["onboarding", "offboarding", "cuti", "leave", "fasilitas",
                         "kantor", "meeting room", "ruang meeting", "atk", "alat tulis",
                         "surat", "dokumen hr", "hrd", "resign", "serah terima",
                         "pengembalian barang", "kembaliin laptop", "pinjam laptop",
                         "peminjaman", "seragam", "id card", "barang keluar"],
    "security": ["phishing", "phising", "malware", "virus", "hack", "diretas", "keamanan",
                 "vulnerability", "suspicious", "penipuan", "mencurigakan", "ransomware",
                 "kebocoran data", "brute force", "scam", "port scan", "akun kena",
                 "security incident", "security alert"],
    "compliance": ["compliance", "audit", "iso 27001", "soc 2", "regulasi", "kepatuhan",
                   "sertifikasi", "policy", "kebijakan", "gdpr", "uu pdp", "pemeriksaan",
                   "bukti audit", "evidence"],
    "account_identity": ["password", "login", "log in", "sign in", "akun", "sso",
                         "2fa", "otp", "verifikasi email", "email kantor", "reset password",
                         "lupa password", "ganti password", "unlock account", "username"],
    "data_reporting": ["laporan", "report", "rekap", "statistik", "analytics", "export data",
                       "ekspor", "dataset", "query data", "minta data", "data untuk",
                       "ambil data", "pull data", "grafik", "excel", "spreadsheet", "csv"],
    "networks_connectivity": ["vpn", "wifi", "wi-fi", "internet", "jaringan", "network",
                              "ping", "dns", "proxy", "konektivitas", "tidak konek",
                              "putus", "remote desktop", "anydesk", "teamviewer", "ip address",
                              "lan "],
    "application_issue": ["error", "gagal", "bug", "crash", "failed", "timeout",
                          "lemot", "slow", "lag", "500", "404", "502", "not working",
                          "tidak jalan", "ngadat", "hang", "blank", "putih", "loading lama",
                          "undefined", "traceback", "exception"],
    "service": ["terima kasih", "thanks", "thank you", "makasih", "maksih", "follow up",
                "followup", "konfirmasi", "selesai", "closing", "bisa di close",
                "di close", "done ya", "sudah ok", "sudah bisa", "update status",
                "update tiket"],
    "request_access": ["minta akses", "akses ke", "akses untuk", "request access",
                       "boleh akses", "buatkan akses", "beri akses", "grant access",
                       "add to", "tambahkan ke", "invite", "whitelist", "registrasi",
                       "daftarkan", "buatkan user", "buat akun", "new credential",
                       "api key", "token baru", "buatkan bucket"],
}

COMPILED = {lab: [(re.compile(r"\b" + re.escape(k.strip()) + r"\b", re.I),
                   2 if " " in k.strip() else 1)
                  for k in kws] for lab, kws in KEYWORDS.items()}


def score(text):
    s = {}
    hits = {}
    for lab, pats in COMPILED.items():
        sc, h = 0, []
        for pat, w in pats:
            n = len(pat.findall(text))
            if n:
                sc += w * min(n, 3)   # cap agar satu kata tak dominan
                h.append(pat.pattern.strip())
        if sc:
            s[lab] = sc
            hits[lab] = h
    return s, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", default=os.path.join(PROC, "tickets_user.jsonl"))
    p.add_argument("--out", default=os.path.join(PROC, "tickets_labeled.jsonl"))
    p.add_argument("--min-score", type=int, default=2)
    p.add_argument("--min-margin", type=int, default=2)
    p.add_argument("--rare-boost", action="store_true",
                   help="tambang pesan belum berlabel utk kelas langka dgn aturan rileks "
                        "(skor>=1, margin>=1 vs kelas non-langka) - hanya utk TRAIN, "
                        "ditandai boost=true; test tetap memakai aturan strict")
    args = p.parse_args()

    RARE = ("security", "compliance", "platforms", "employee_support", "data_reporting")
    rows = [json.loads(l) for l in open(args.in_file, encoding="utf-8")]
    out, dist = [], Counter()
    per_lang = defaultdict(Counter)
    strict_texts = set()
    for r in rows:
        s, hits = score(r["text"])
        if not s:
            continue
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        top, top_sc = ranked[0]
        second_sc = ranked[1][1] if len(ranked) > 1 else 0
        if top_sc < args.min_score or (top_sc - second_sc) < args.min_margin:
            continue
        out.append({"text": r["text"], "label": top, "score": top_sc,
                    "ticket_id": r["ticket_id"], "source": r["source"],
                    "lang_guess": r["lang_guess"], "hits": hits[top][:4]})
        strict_texts.add(r["text"])
        dist[top] += 1
        per_lang[r["lang_guess"]][top] += 1

    n_boost = 0
    if args.rare_boost:
        for r in rows:
            if r["text"] in strict_texts:
                continue
            s, hits = score(r["text"])
            rare = {k: v for k, v in s.items() if k in RARE}
            if not rare:
                continue
            top, top_sc = max(rare.items(), key=lambda kv: kv[1])
            other = max((v for k, v in s.items() if k not in RARE), default=0)
            if top_sc >= 1 and (top_sc - other) >= 1:
                out.append({"text": r["text"], "label": top, "score": top_sc,
                            "ticket_id": r["ticket_id"], "source": r["source"],
                            "lang_guess": r["lang_guess"], "hits": hits[top][:4],
                            "boost": True})
                dist[top] += 1
                per_lang[r["lang_guess"]][top] += 1
                n_boost += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"berlabel: {len(out)}/{len(rows)} pesan ({len(out)/len(rows)*100:.1f}%)"
          + (f" [termasuk {n_boost} boost kelas langka]" if n_boost else ""))
    for lab, n in dist.most_common():
        print(f"  {lab:<22} {n:>6}  ({n/len(out)*100:.1f}%)")
    with open(os.path.join(PROC, "tickets_label_dist.json"), "w", encoding="utf-8") as f:
        json.dump({"total": len(out), "dist": dict(dist),
                   "per_lang": {k: dict(v) for k, v in per_lang.items()},
                   "labels": LABELS}, f, ensure_ascii=False, indent=2)
    print(f"tersimpan: {args.out}")


if __name__ == "__main__":
    main()
