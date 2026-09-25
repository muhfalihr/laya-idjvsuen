"""Ambil seluruh rows dari tabel pesan tiket pada dua database internal
(PostgreSQL + MySQL), normalisasi ke skema sama, tulis JSONL mentah.

Kredensial dibaca dari secrets/db.ini (gitignored; contoh: db.example.ini).

Output: data/raw/tickets/ticket_messages_raw.jsonl
  {source, db_id, ticket_id, user_id, username, message, message_from, timestamp, chat_id}
"""
import argparse
import configparser
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRETS = os.path.join(ROOT, "secrets", "db.ini")
OUT_DIR = os.path.join(ROOT, "data", "raw", "tickets")


def load_cfg():
    cp = configparser.ConfigParser()
    if not cp.read(SECRETS):
        raise SystemExit(f"tidak ada {SECRETS} - salin dari secrets/db.example.ini lalu isi.")
    return cp


def fetch_pg(cp):
    import psycopg2
    s = cp["postgres"]
    conn = psycopg2.connect(host=s["host"], port=int(s["port"]), dbname=s["db"],
                            user=s["user"], password=s["password"], connect_timeout=10)
    rows = []
    with conn.cursor(name="fetch_ticket_messages") as cur:  # server-side cursor: 100k rows aman
        cur.itersize = 5000
        cur.execute("""SELECT id, ticket_id, user_id, COALESCE(username, ''),
                              COALESCE(message, ''), COALESCE(message_from, ''),
                              COALESCE(timestamp::text, ''), COALESCE(message_chat_id::text, '')
                       FROM ticket_messages""")
        for r in cur:
            rows.append({"source": "postgres", "db_id": r[0], "ticket_id": r[1],
                         "user_id": str(r[2]), "username": r[3], "message": r[4],
                         "message_from": r[5], "timestamp": r[6], "chat_id": r[7]})
    conn.close()
    return rows


def fetch_mysql(cp):
    import pymysql
    s = cp["mysql"]
    conn = pymysql.connect(host=s["host"], port=int(s["port"]), database=s["db"],
                           user=s["user"], password=s["password"], connect_timeout=10)
    rows = []
    with conn.cursor() as cur:
        cur.execute("""SELECT id, ticket_id, user_id, COALESCE(username, ''),
                              COALESCE(message, ''), COALESCE(message_from, ''),
                              COALESCE(timestamp, ' '), COALESCE(message_chat_id, '')
                       FROM ticket_messages""")
        for r in cur.fetchall():
            rows.append({"source": "mysql", "db_id": r[0], "ticket_id": str(r[1]),
                         "user_id": str(r[2]), "username": str(r[3]), "message": str(r[4]),
                         "message_from": str(r[5]), "timestamp": str(r[6]), "chat_id": str(r[7])})
    conn.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", choices=["pg", "mysql"], default=["pg", "mysql"])
    args = p.parse_args()

    cp = load_cfg()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ticket_messages_raw.jsonl")

    all_rows = []
    if "pg" in args.sources:
        rows = fetch_pg(cp)
        print(f"[postgres] {len(rows)} rows")
        all_rows.extend(rows)
    if "mysql" in args.sources:
        rows = fetch_mysql(cp)
        print(f"[mysql] {len(rows)} rows")
        all_rows.extend(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
