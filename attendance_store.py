import json
import sqlite3
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
ATTENDANCE_DB = DATA_DIR / "rollcall.db"


def init_attendance_db(db_path=ATTENDANCE_DB):
    """Create or migrate the local attendance history database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'success',
                image_path TEXT,
                face_score REAL,
                face_confidence REAL,
                fingerprint_score REAL,
                decision REAL,
                note TEXT
            )
        """)

        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(attendance)")
        }
        migrations = {
            "status": "ALTER TABLE attendance ADD COLUMN status TEXT NOT NULL DEFAULT 'success'",
            "image_path": "ALTER TABLE attendance ADD COLUMN image_path TEXT",
            "face_score": "ALTER TABLE attendance ADD COLUMN face_score REAL",
            "face_confidence": "ALTER TABLE attendance ADD COLUMN face_confidence REAL",
            "fingerprint_score": "ALTER TABLE attendance ADD COLUMN fingerprint_score REAL",
            "decision": "ALTER TABLE attendance ADD COLUMN decision REAL",
            "note": "ALTER TABLE attendance ADD COLUMN note TEXT",
        }

        for column, sql in migrations.items():
            if column not in existing_columns:
                conn.execute(sql)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                image_path TEXT,
                embedding TEXT NOT NULL,
                face_score REAL,
                face_confidence REAL,
                fingerprint_position INTEGER NOT NULL DEFAULT -1,
                status TEXT NOT NULL DEFAULT 'pending',
                approved_name TEXT,
                note TEXT
            )
        """)
        existing_pending_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(pending_faces)")
        }
        if "fingerprint_position" not in existing_pending_columns:
            conn.execute(
                "ALTER TABLE pending_faces "
                "ADD COLUMN fingerprint_position INTEGER NOT NULL DEFAULT -1"
            )
        conn.commit()
    finally:
        conn.close()


def record_attendance(
    name,
    status,
    image_path=None,
    face_score=None,
    face_confidence=None,
    fingerprint_score=None,
    decision=None,
    note=None,
    db_path=ATTENDANCE_DB,
):
    """Persist one attendance/verification event for the dashboard."""
    init_attendance_db(db_path)
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO attendance (
                name, time, status, image_path, face_score, face_confidence,
                fingerprint_score, decision, note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                event_time,
                status,
                image_path,
                face_score,
                face_confidence,
                fingerprint_score,
                decision,
                note,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_pending_face(
    embedding,
    image_path=None,
    face_score=None,
    face_confidence=None,
    fingerprint_position=-1,
    note=None,
    db_path=ATTENDANCE_DB,
):
    """Persist an unknown face so the dashboard can approve it later."""
    init_attendance_db(db_path)
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    embedding_json = json.dumps(embedding.tolist())

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO pending_faces (
                time, image_path, embedding, face_score, face_confidence,
                fingerprint_position, note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_time,
                image_path,
                embedding_json,
                face_score,
                face_confidence,
                fingerprint_position,
                note,
            ),
        )
        conn.commit()
    finally:
        conn.close()
