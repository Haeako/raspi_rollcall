import json
import numpy as np
import os
import sqlite3
import sys
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

from attendance_store import ATTENDANCE_DB, CAPTURES_DIR, init_attendance_db


app = Flask(__name__)
DB_PATH = ATTENDANCE_DB
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "app" / "config.json"

pipeline_thread = None
pipeline_lock = threading.Lock()


def row_to_dict(row):
    return dict(row) if row else None


def get_db_connection():
    init_attendance_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_attendance_data(limit=50):
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id, name, time, status, image_path, face_score,
                    face_confidence, fingerprint_score, decision, note
                FROM attendance
                ORDER BY time DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [row_to_dict(row) for row in rows]
    except Exception as e:
        print("Loi ket noi DB:", e)
        return []


def get_attendance_stats():
    empty_stats = {
        "total": 0,
        "success": 0,
        "pending_unknown_face": 0,
        "manual_review": 0,
        "uncertain": 0,
        "denied": 0,
        "pending_faces": 0,
    }

    try:
        with get_db_connection() as conn:
            attendance_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM attendance
                GROUP BY status
                """
            ).fetchall()
            pending_faces = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM pending_faces
                WHERE status = 'pending'
                """
            ).fetchone()

        stats = empty_stats.copy()
        for row in attendance_rows:
            status = row["status"]
            count = int(row["count"])
            stats["total"] += count
            if status in stats:
                stats[status] = count
            else:
                stats["denied"] += count

        stats["pending_faces"] = int(pending_faces["count"]) if pending_faces else 0
        stats["needs_review"] = (
            stats["manual_review"]
            + stats["uncertain"]
            + stats["pending_unknown_face"]
            + stats["pending_faces"]
        )
        return stats
    except Exception as e:
        print("Loi thong ke DB:", e)
        stats = empty_stats.copy()
        stats["needs_review"] = 0
        return stats


def get_attendance_status():
    try:
        with get_db_connection() as conn:
            latest = conn.execute(
                """
                SELECT
                    id, name, time, status, image_path, face_score,
                    face_confidence, fingerprint_score, decision, note
                FROM attendance
                ORDER BY time DESC, id DESC
                LIMIT 1
                """
            ).fetchone()

        if not latest:
            return {
                "state": "idle",
                "label": "San sang",
                "message": "Chua co luot diem danh nao.",
                "name": "-",
                "time": "-",
                "status": "idle",
                "status_label": "Cho kich hoat",
                "badge_class": "bg-secondary",
                "icon": "fa-circle-dot",
                "latest_id": None,
                "note": "",
            }

        item = row_to_dict(latest)
        status = item.get("status")
        status_meta = {
            "success": ("Thanh cong", "bg-success", "fa-circle-check"),
            "pending_unknown_face": ("Cho duyet", "bg-warning text-dark", "fa-user-clock"),
            "manual_review": ("Can xac minh", "bg-warning text-dark", "fa-triangle-exclamation"),
            "uncertain": ("Can xac minh", "bg-warning text-dark", "fa-triangle-exclamation"),
            "denied": ("Tu choi", "bg-danger", "fa-circle-xmark"),
        }
        label, badge_class, icon = status_meta.get(
            status,
            ("Tu choi", "bg-danger", "fa-circle-xmark"),
        )

        if status == "success":
            message = f"Da ghi nhan thanh cong cho {item['name']}."
            state = "ok"
        elif status in {"manual_review", "uncertain", "pending_unknown_face"}:
            message = "Luot moi can nguoi quan tri xu ly."
            state = "review"
        else:
            message = "Luot moi bi tu choi."
            state = "denied"

        item.update(
            {
                "state": state,
                "label": label,
                "message": message,
                "status_label": label,
                "badge_class": badge_class,
                "icon": icon,
                "latest_id": item["id"],
            }
        )
        return item
    except Exception as e:
        print("Loi doc trang thai diem danh:", e)
        return {
            "state": "error",
            "label": "Loi he thong",
            "message": "Khong doc duoc trang thai diem danh.",
            "name": "-",
            "time": "-",
            "status": "error",
            "status_label": "Loi",
            "badge_class": "bg-danger",
            "icon": "fa-triangle-exclamation",
            "latest_id": None,
            "note": str(e),
        }


def load_qdrant_config():
    cfg = {"qdrant_host": "qdrant", "qdrant_port": 6333}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg["qdrant_host"] = os.getenv("QDRANT_HOST", cfg["qdrant_host"])
    cfg["qdrant_port"] = int(os.getenv("QDRANT_PORT", cfg["qdrant_port"]))
    return cfg


def load_qdrant_db_class():
    core_src = PROJECT_ROOT / "core" / "src"
    core_src_path = str(core_src)
    if core_src_path not in sys.path:
        sys.path.insert(0, core_src_path)

    from database import Qdrant_db

    return Qdrant_db


def delete_fingerprint_template(position):
    if position is None or position < 0:
        return {"success": True, "message": "No fingerprint template to delete"}

    core_src = PROJECT_ROOT / "core" / "src"
    core_src_path = str(core_src)
    if core_src_path not in sys.path:
        sys.path.insert(0, core_src_path)

    from AS608 import AS608_HAL

    sensor = AS608_HAL()
    result = sensor.delete(position)
    return {
        "success": result.success,
        "message": result.message,
        "position": result.position,
    }


def get_pending_faces(limit=20):
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.id, p.time, p.image_path, p.face_score,
                    p.face_confidence, p.fingerprint_position, p.note,
                    (
                        SELECT a.id
                        FROM attendance a
                        WHERE a.image_path = p.image_path
                        ORDER BY a.id DESC
                        LIMIT 1
                    ) AS attendance_id
                FROM pending_faces p
                WHERE p.status = 'pending'
                ORDER BY p.time DESC, p.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            pending = [row_to_dict(row) for row in rows]
            for item in pending:
                if item.get("image_path"):
                    item["image_url"] = url_for(
                        "capture_image",
                        filename=item["image_path"],
                    )
                else:
                    item["image_url"] = None
            return pending
    except Exception as e:
        print("Loi doc pending faces:", e)
        return []


def get_attendance_detail(attendance_id):
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT
                id, name, time, status, image_path, face_score,
                face_confidence, fingerprint_score, decision, note
            FROM attendance
            WHERE id = ?
            """,
            (attendance_id,),
        ).fetchone()

    detail = row_to_dict(row)
    if not detail:
        return None

    if detail.get("image_path"):
        detail["image_url"] = url_for(
            "capture_image",
            filename=detail["image_path"],
        )
        with get_db_connection() as conn:
            pending = conn.execute(
                """
                SELECT id
                FROM pending_faces
                WHERE image_path = ? AND status = 'pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (detail["image_path"],),
            ).fetchone()
        detail["pending_id"] = pending["id"] if pending else None
    else:
        detail["image_url"] = None
        detail["pending_id"] = None
    return detail


def start_pipeline_once():
    global pipeline_thread

    with pipeline_lock:
        if pipeline_thread and pipeline_thread.is_alive():
            return

        pipeline_thread = threading.Thread(
            target=run_pipeline,
            name="RollcallPipeline",
            daemon=True,
        )
        pipeline_thread.start()


def run_pipeline():
    try:
        from app.app import main as run_rollcall_pipeline

        run_rollcall_pipeline()
    except Exception as e:
        print("Pipeline backend stopped:", e)


@app.route("/")
def index():
    data = get_attendance_data()
    pending_faces = get_pending_faces()
    stats = get_attendance_stats()
    attendance_status = get_attendance_status()
    return render_template(
        "index.html",
        data=data,
        pending_faces=pending_faces,
        stats=stats,
        attendance_status=attendance_status,
    )


@app.route("/status-screen")
def status_screen():
    return render_template("status_screen.html")


@app.route("/idle-screen")
def idle_screen():
    return render_template("idle_screen.html")


@app.route("/api/stats")
def attendance_stats():
    return jsonify(get_attendance_stats())


@app.route("/api/status")
def attendance_status():
    return jsonify(get_attendance_status())


@app.route("/api/attendance/<int:attendance_id>")
def attendance_detail(attendance_id):
    detail = get_attendance_detail(attendance_id)
    if not detail:
        abort(404)
    return jsonify(detail)


@app.route("/api/attendance/<int:attendance_id>/approve", methods=["POST"])
def approve_attendance_review(attendance_id):
    with get_db_connection() as conn:
        result = conn.execute(
            """
            UPDATE attendance
            SET status = 'success',
                note = 'Manual review approved from dashboard'
            WHERE id = ?
              AND (
                status = 'manual_review'
                OR (status = 'uncertain' AND name != 'Unknown')
              )
            """,
            (attendance_id,),
        )

        if result.rowcount == 0:
            abort(404)

    return jsonify({"success": True})


@app.route("/api/attendance/<int:attendance_id>/deny", methods=["POST"])
def deny_attendance_review(attendance_id):
    with get_db_connection() as conn:
        result = conn.execute(
            """
            UPDATE attendance
            SET status = 'denied',
                note = 'Manual review denied from dashboard'
            WHERE id = ?
              AND (
                status = 'manual_review'
                OR (status = 'uncertain' AND name != 'Unknown')
              )
            """,
            (attendance_id,),
        )

        if result.rowcount == 0:
            abort(404)

    return jsonify({"success": True})


@app.route("/api/pending_faces/<int:pending_id>/approve", methods=["POST"])
def approve_pending_face(pending_id):
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, embedding, image_path, fingerprint_position
            FROM pending_faces
            WHERE id = ? AND status = 'pending'
            """,
            (pending_id,),
        ).fetchone()

        if not row:
            abort(404)

        try:
            fingerprint_position = int(
                payload.get("fingerprint_position", row["fingerprint_position"])
            )
        except (TypeError, ValueError):
            fingerprint_position = int(row["fingerprint_position"] or -1)

        embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)
        cfg = load_qdrant_config()
        Qdrant_db = load_qdrant_db_class()

        db = Qdrant_db(
            host=cfg["qdrant_host"],
            port=int(cfg["qdrant_port"]),
        )
        db.add(
            embedding,
            name,
            fingerprint_position=fingerprint_position,
        )

        conn.execute(
            """
            UPDATE pending_faces
            SET status = 'approved', approved_name = ?
            WHERE id = ?
            """,
            (name, pending_id),
        )

        if row["image_path"]:
            conn.execute(
                """
                UPDATE attendance
                SET name = ?,
                    status = 'success',
                    note = 'Approved from dashboard'
                WHERE image_path = ?
                  AND (
                    status = 'pending_unknown_face'
                    OR (status = 'uncertain' AND name = 'Unknown')
                  )
                """,
                (name, row["image_path"]),
            )

    return jsonify({"success": True, "name": name})


@app.route("/api/pending_faces/<int:pending_id>/skip", methods=["POST"])
def skip_pending_face(pending_id):
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT image_path, fingerprint_position
            FROM pending_faces
            WHERE id = ? AND status = 'pending'
            """,
            (pending_id,),
        ).fetchone()

        if not row:
            abort(404)

        delete_result = delete_fingerprint_template(int(row["fingerprint_position"]))
        if not delete_result["success"]:
            return jsonify({
                "success": False,
                "error": delete_result["message"],
            }), 500

        result = conn.execute(
            """
            UPDATE pending_faces
            SET status = 'skipped', note = 'Skipped from dashboard'
            WHERE id = ? AND status = 'pending'
            """,
            (pending_id,),
        )

        if result.rowcount == 0:
            abort(404)

        if row["image_path"]:
            conn.execute(
                """
                UPDATE attendance
                SET status = 'denied',
                    note = 'Skipped from dashboard'
                WHERE image_path = ?
                  AND (
                    status = 'pending_unknown_face'
                    OR (status = 'uncertain' AND name = 'Unknown')
                  )
                """,
                (row["image_path"],),
            )

    return jsonify({"success": True})


@app.route("/captures/<path:filename>")
def capture_image(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    init_attendance_db(DB_PATH)
    start_pipeline_once()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
