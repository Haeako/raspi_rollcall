#!/usr/bin/env python3
"""Clean-reset rollcall data, face database, and fingerprint templates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from attendance_store import ATTENDANCE_DB, CAPTURES_DIR, init_attendance_db
from core import AS608_HAL, Qdrant_db, get_config_path


def load_qdrant_config() -> tuple[str, int]:
    config_path = get_config_path()
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    return (
        str(os.getenv("QDRANT_HOST", config.get("qdrant_host", "localhost"))),
        int(os.getenv("QDRANT_PORT", config.get("qdrant_port", 6333))),
    )


def confirm_or_exit(skip_confirm: bool) -> None:
    if skip_confirm:
        return

    print("CANH BAO: thao tac nay se xoa sach:")
    print("- SQLite attendance/pending database")
    print("- Anh trong data/captures")
    print("- Qdrant face collection")
    print("- Tat ca template van tay tren AS608")
    answer = input("Go CLEAN RESET? nhap 'RESET' de tiep tuc: ").strip()
    if answer != "RESET":
        print("Da huy.")
        raise SystemExit(1)


def reset_sqlite_and_captures() -> None:
    if ATTENDANCE_DB.exists():
        ATTENDANCE_DB.unlink()
        print(f"[OK] Deleted SQLite DB: {ATTENDANCE_DB}")
    else:
        print(f"[SKIP] SQLite DB not found: {ATTENDANCE_DB}")

    if CAPTURES_DIR.exists():
        shutil.rmtree(CAPTURES_DIR)
        print(f"[OK] Deleted captures: {CAPTURES_DIR}")
    else:
        print(f"[SKIP] Captures dir not found: {CAPTURES_DIR}")

    init_attendance_db()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    print("[OK] Recreated empty attendance schema and captures dir")


def reset_qdrant(host: str | None = None, port: int | None = None) -> None:
    config_host, config_port = load_qdrant_config()
    host = host or config_host
    port = port or config_port
    candidates = [(host, port)]
    if host in {"localhost", "127.0.0.1"}:
        candidates.append(("qdrant", port))

    last_error = None
    for candidate_host, candidate_port in candidates:
        try:
            db = Qdrant_db(host=candidate_host, port=candidate_port)
            db.clear()
            print(
                f"[OK] Cleared Qdrant collection "
                f"'{db.collection_name}' at {candidate_host}:{candidate_port}"
            )
            return
        except urllib.error.URLError as error:
            last_error = error
            print(f"[WARN] Cannot connect Qdrant at {candidate_host}:{candidate_port}: {error.reason}")

    raise RuntimeError(
        "Khong ket noi duoc Qdrant. Hay start service bang "
        "`docker compose -f compose/docker-compose.yml up -d qdrant`, "
        "hoac chay lai voi `--skip-qdrant`, "
        "hoac truyen dung host bang `--qdrant-host qdrant`."
    ) from last_error


def reset_fingerprints() -> None:
    sensor = AS608_HAL()
    result = sensor.empty()
    if not result.success:
        raise RuntimeError(result.message)
    print("[OK] Cleared all AS608 fingerprint templates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean reset rollcall SQLite, captures, Qdrant, and AS608.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="Do not clear AS608 fingerprint templates.",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Do not clear Qdrant face collection.",
    )
    parser.add_argument(
        "--qdrant-host",
        help="Override Qdrant host from config/env.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        help="Override Qdrant port from config/env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confirm_or_exit(args.yes)

    reset_sqlite_and_captures()

    if args.skip_qdrant:
        print("[SKIP] Qdrant reset")
    else:
        reset_qdrant(args.qdrant_host, args.qdrant_port)

    if args.skip_fingerprint:
        print("[SKIP] AS608 fingerprint reset")
    else:
        reset_fingerprints()

    print("[DONE] Clean reset complete.")


if __name__ == "__main__":
    main()
