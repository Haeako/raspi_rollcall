"""Synchronous face + fingerprint rollcall pipeline."""

import sys
import time
import json
import logging
import os
import queue
from datetime import datetime
from enum import Enum, auto
from multiprocessing import Process, Queue, Event
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    PiCamera,
    Qdrant_db,
    FaceModel,
    FuzzyModel,
    AS608_HAL,
    HW_201_HAL,
    get_config_path,
    ensure_capture_dir,
)
from attendance_store import init_attendance_db, record_attendance, record_pending_face

class PipelineState(Enum):
    IDLE = auto()
    WAIT_FACE = auto()
    WAIT_FP = auto()
    VERIFY = auto()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s] %(message)s",
)

logger = logging.getLogger(__name__)

CONFIG_PATH = get_config_path()

DEFAULT_CONFIG = {
    "trigger": "sensor",

    # Sensor
    "sensor_pin": 26,
    "sensor_cooldown": 3.0,

    # Face
    "camera_width": 320,
    "camera_height": 240,
    "camera_framerate": 10,
    "camera_buffer_count": 2,
    "face_confidence_thresh": 0.5,
    "face_threshold": 0.4,

    # Timeout
    "face_timeout": 5.0,
    "fp_timeout": 10.0,
    "worker_startup_timeout": 90.0,

    # Weights (just filenames, paths resolved by core)
    "det_weight": "det_10g.onnx",
    "rec_weight": "w600k_r50.onnx",

    # Qdrant
    "qdrant_host": "qdrant",
    "qdrant_port": 6333,
}

VALID_TRIGGERS = {"sensor"}


def save_attendance_capture(frame, bboxes=None, name="Unknown"):
    """Save the captured frame used for face verification."""
    import cv2

    capture_dir = ensure_capture_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    capture_path = capture_dir / filename

    annotated = frame.copy()

    if bboxes is not None and len(bboxes) > 0:
        for bb in bboxes:
            left, top, right, bottom = [int(x) for x in bb]
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 180, 0), 2)
            cv2.putText(
                annotated,
                name,
                (max(left, 8), max(top - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 180, 0),
                2,
            )

    cv2.imwrite(str(capture_path), annotated)
    return filename


def orient_frame(frame):
    """Rotate camera frame 180 degrees before recognition and storage."""
    return frame[::-1, ::-1].copy()


def load_config(path=CONFIG_PATH):
    """Load config from file, merging with defaults."""
    cfg = DEFAULT_CONFIG.copy()

    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
            logger.info(f"Config loaded: {path}")
        except Exception as e:
            logger.warning(f"Load config failed: {e}, using defaults")
    else:
        logger.info("No config file found, using defaults")

    cfg["trigger"] = str(cfg.get("trigger", DEFAULT_CONFIG["trigger"])).lower()
    cfg["qdrant_host"] = os.getenv("QDRANT_HOST", cfg["qdrant_host"])
    cfg["qdrant_port"] = int(os.getenv("QDRANT_PORT", cfg["qdrant_port"]))

    if cfg["trigger"] not in VALID_TRIGGERS:
        logger.warning("Invalid trigger=%r, forcing 'sensor'", cfg["trigger"])
        cfg["trigger"] = "sensor"

    return cfg


def clear_queue(q):
    """Clear a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def request_face_capture(request_event, response_queue, logger):
    """Request face capture from worker."""
    clear_queue(response_queue)
    logger.info("Requesting face capture...")
    request_event.set()
    return time.time()


def request_fingerprint(request_queue, response_queue, logger, action="search"):
    """Request a fingerprint worker action."""
    clear_queue(request_queue)
    clear_queue(response_queue)
    logger.info("Requesting fingerprint %s...", action)
    request_queue.put({"action": action})
    return time.time()


def reset_session():
    """Return clean per-scan state."""
    return {
        "face": None,
        "fingerprint": None,
        "face_requested_at": 0.0,
        "fp_requested_at": 0.0,
    }


def set_state(current_state, next_state, reason=""):
    """Move state machine to the next state and log transitions."""
    if current_state != next_state:
        suffix = f" ({reason})" if reason else ""
        logger.info("STATE %s -> %s%s", current_state.name, next_state.name, suffix)
    return next_state


def status(title, *lines):
    message = " | ".join(str(line) for line in lines if line)
    if message:
        logger.info("STATUS %s | %s", title, message)
    else:
        logger.info("STATUS %s", title)


def face_worker(request_event, response_queue, stop_event, ready_event, cfg):
    """Face detection and recognition worker process."""
    logger.info("Face worker started.")

    try:
        camera = PiCamera(
            width=int(cfg["camera_width"]),
            height=int(cfg["camera_height"]),
            framerate=int(cfg["camera_framerate"]),
            buffer_count=int(cfg["camera_buffer_count"]),
        )
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        return

    try:
        model = FaceModel(
            det_weight=cfg["det_weight"],
            rec_weight=cfg["rec_weight"],
            confidence_thresh=cfg["face_confidence_thresh"],
        )
    except Exception as e:
        logger.error(f"Face model initialization failed: {e}")
        return

    try:
        db = Qdrant_db(
            host=cfg["qdrant_host"],
            port=cfg["qdrant_port"],
        )
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return

    threshold = cfg["face_threshold"]
    ready_event.set()
    logger.info("Face worker ready.")

    try:
        while not stop_event.is_set():
            if not request_event.wait(timeout=0.1):
                continue

            request_event.clear()

            try:
                frame = orient_frame(camera.capture_array())
                embs, bbs, confs = model.get_embeding_vector(
                    frame,
                    image_format="RGB",
                )

                if len(embs) == 0:
                    response_queue.put({
                        "success": False,
                        "message": "No face detected",
                    })
                    continue

                best_idx = int(confs.argmax()) if hasattr(confs, "argmax") else 0
                best_emb = embs[best_idx]
                best_conf = float(confs[best_idx])

                results = db.search(best_emb, top_k=1, threshold=threshold)

                if results:
                    name, score, fp_pos = results[0]
                else:
                    name, score, fp_pos = "Unknown", 0.0, -1

                capture_path = save_attendance_capture(
                    frame,
                    bboxes=[bbs[best_idx]],
                    name=name,
                )

                response_queue.put({
                    "success": True,
                    "name": name,
                    "score": score,
                    "confidence": best_conf,
                    "fp_pos": fp_pos,
                    "capture_path": capture_path,
                    "embedding": best_emb,
                    "ts": time.time(),
                })

            except Exception as e:
                logger.error(f"Face processing error: {e}")
                response_queue.put({
                    "success": False,
                    "message": str(e),
                })

    finally:
        camera.close()
        logger.info("Face worker stopped.")


def fingerprint_worker(request_queue, response_queue, stop_event, ready_event, cfg):
    """Fingerprint scanning worker process."""
    logger.info("Fingerprint worker started.")

    try:
        sensor = AS608_HAL()
    except RuntimeError as e:
        logger.error(f"AS608 initialization failed: {e}")
        return

    ready_event.set()
    logger.info("Fingerprint worker ready.")

    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            action = request.get("action", "search")
            try:
                if action == "enroll":
                    result = sensor.enroll(timeout=cfg["fp_timeout"])
                elif action == "delete":
                    result = sensor.delete(int(request.get("position", -1)))
                else:
                    action = "search"
                    result = sensor.search(timeout=cfg["fp_timeout"])

                if result.success:
                    response_queue.put({
                        "success": True,
                        "action": action,
                        "position": result.position,
                        "score": result.score,
                        "message": result.message,
                        "ts": time.time(),
                    })
                else:
                    response_queue.put({
                        "success": False,
                        "action": action,
                        "message": result.message,
                        "position": result.position,
                        "score": result.score,
                    })

            except Exception as e:
                logger.error(f"Fingerprint processing error: {e}")
                response_queue.put({
                    "success": False,
                    "action": action,
                    "message": str(e),
                })

    finally:
        logger.info("Fingerprint worker stopped.")


def main():
    """Main pipeline orchestrator."""
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    cfg = load_config()
    init_attendance_db()
    logger.info("Pipeline initializing...")
    logger.info("Flow: HW-201 -> face -> fingerprint -> fuzzy -> result")

    trigger = cfg["trigger"]
    sensor = None

    if trigger == "sensor":
        try:
            sensor = HW_201_HAL(sensor_pin=cfg["sensor_pin"])
            logger.info("HW-201 sensor ready.")
        except Exception as e:
            logger.error(f"HW-201 initialization failed: {e}")
            status("ERROR", "Khong khoi tao duoc cam bien", str(e))
            return

    stop_event = Event()

    face_request = Event()
    face_response = Queue(maxsize=1)
    face_ready = Event()

    fp_request = Queue(maxsize=1)
    fp_response = Queue(maxsize=1)
    fp_ready = Event()

    try:
        db = Qdrant_db(
            host=cfg["qdrant_host"],
            port=cfg["qdrant_port"],
        )
        logger.info("Qdrant database connected.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        status("ERROR", "Khong ket noi duoc database", str(e))
        return

    fuzzy = None
    try:
        fuzzy = FuzzyModel()
        logger.info("Fuzzy model initialized.")
    except Exception as e:
        logger.error(f"Fuzzy model initialization failed: {e}")
        status("ERROR", "Khong khoi tao duoc fuzzy model", str(e))
        return

    try:
        ensure_capture_dir()
    except Exception as e:
        logger.warning(f"Could not create captures directory: {e}")

    processes = []

    processes.append(Process(
        target=face_worker,
        args=(face_request, face_response, stop_event, face_ready, cfg),
        daemon=True,
        name="FaceWorker",
    ))

    processes.append(Process(
        target=fingerprint_worker,
        args=(fp_request, fp_response, stop_event, fp_ready, cfg),
        daemon=True,
        name="FingerprintWorker",
    ))

    for p in processes:
        p.start()
        logger.info(f"Started {p.name} (PID={p.pid})")

    required_workers = [
        ("FaceWorker", face_ready),
        ("FingerprintWorker", fp_ready),
    ]

    startup_deadline = time.time() + cfg["worker_startup_timeout"]
    for worker_name, ready_event in required_workers:
        remaining = max(0.0, startup_deadline - time.time())
        logger.info("Waiting for %s to be ready...", worker_name)
        if not ready_event.wait(timeout=remaining):
            logger.error(
                "%s did not become ready within %.1fs; stopping pipeline.",
                worker_name,
                cfg["worker_startup_timeout"],
            )
            stop_event.set()
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            return
        logger.info("%s is ready.", worker_name)

    cooldown = cfg["sensor_cooldown"]
    last_trigger = 0

    logger.info("✅ Pipeline running. Press Ctrl+C to stop.")
    status("READY", "Dat nguoi vao cam bien", "Cho kich hoat")

    state = PipelineState.IDLE
    session = reset_session()

    try:
        while True:
            if state == PipelineState.IDLE:
                if not sensor.detect():
                    now = time.time()
                    time.sleep(0.05)
                    continue

                now = time.time()
                if now - last_trigger < cooldown:
                    time.sleep(0.05)
                    continue

                last_trigger = now
                logger.info("📍 Sensor triggered")
                status("SCANNING", "Da phat hien nguoi", "Dang quet khuon mat")
                session = reset_session()
                session["face_requested_at"] = request_face_capture(face_request, face_response, logger)
                state = set_state(state, PipelineState.WAIT_FACE, "sensor trigger")

            elif state == PipelineState.WAIT_FACE:
                try:
                    face_data = face_response.get_nowait()
                    
                    if not face_data["success"]:
                        logger.warning(f"⚠️ Face detection failed: {face_data['message']}")
                        status("FACE FAILED", "Khong nhan dien duoc mat", face_data["message"])
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "face failed")
                        continue

                    session["face"] = face_data
                    face_name = face_data["name"]
                    logger.info(f"👤 Face detected: {face_name}")

                    if face_name == "Unknown":
                        logger.info("Unknown face; enrolling fingerprint before dashboard decision")
                        status("PENDING", "Khuon mat moi", "Dang enroll van tay")
                        session["fp_requested_at"] = request_fingerprint(
                            fp_request,
                            fp_response,
                            logger,
                            "enroll",
                        )
                        state = set_state(state, PipelineState.WAIT_FP, "unknown face")
                    else:
                        logger.info(f"👋 Hello {face_name}")
                        status("FACE OK", face_name, "Dang quet van tay")
                        session["fp_requested_at"] = request_fingerprint(
                            fp_request,
                            fp_response,
                            logger,
                        )
                        state = set_state(state, PipelineState.WAIT_FP, "face matched")

                except queue.Empty:
                    if time.time() - session["face_requested_at"] > cfg["face_timeout"]:
                        logger.warning("⏱️ Face detection timeout")
                        status("TIMEOUT", "Qua thoi gian quet mat", "Thu lai")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "face timeout")
                    else:
                        time.sleep(0.05)

            elif state == PipelineState.WAIT_FP:
                try:
                    fp_data = fp_response.get_nowait()
                    face_data = session["face"]
                    
                    if not fp_data["success"]:
                        logger.warning(f"⚠️ Fingerprint failed: {fp_data['message']}")
                        status("FINGERPRINT FAILED", "Khong khop van tay", fp_data["message"])
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "fingerprint failed")
                        continue

                    session["fingerprint"] = fp_data
                    if face_data and face_data["name"] == "Unknown":
                        fp_pos = fp_data.get("position", -1)
                        status("PENDING", "Da enroll van tay", "Cho duyet tren dashboard")
                        record_pending_face(
                            embedding=face_data["embedding"],
                            image_path=face_data.get("capture_path"),
                            face_score=face_data.get("score"),
                            face_confidence=face_data.get("confidence"),
                            fingerprint_position=fp_pos,
                            note="Waiting for dashboard approval",
                        )
                        record_attendance(
                            name="Unknown",
                            status="pending_unknown_face",
                            image_path=face_data.get("capture_path"),
                            face_score=face_data.get("score"),
                            face_confidence=face_data.get("confidence"),
                            fingerprint_score=fp_data.get("score"),
                            note=f"Unknown face waiting for dashboard approval; fingerprint_position={fp_pos}",
                        )
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "unknown face pending")
                        continue

                    status("VERIFYING", "Van tay da khop", "Dang xac minh")
                    state = set_state(state, PipelineState.VERIFY, "fingerprint matched")

                except queue.Empty:
                    if time.time() - session["fp_requested_at"] > cfg["fp_timeout"]:
                        logger.warning("⏱️ Fingerprint scan timeout")
                        status("TIMEOUT", "Qua thoi gian quet van tay", "Thu lai")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "fingerprint timeout")
                    else:
                        time.sleep(0.05)

            elif state == PipelineState.VERIFY:
                face_data = session["face"]
                fp_data = session["fingerprint"]

                fp_pos = fp_data["position"]
                fp_name = db.get_name_by_fp_position(fp_pos)
                fp_score = fp_data["score"]

                logger.info(f"🔍 Fingerprint: {fp_name} (score={fp_score})")

                face_name = face_data["name"]
                face_conf = face_data["confidence"]

                if fp_name != face_name:
                    logger.warning(f"❌ BIOMETRIC MISMATCH (face={face_name}, fp={fp_name})")
                    status("DENIED", face_name, f"Van tay thuoc {fp_name}")
                    record_attendance(
                        name=face_name,
                        status="denied",
                        image_path=face_data.get("capture_path"),
                        face_score=face_data.get("score"),
                        face_confidence=face_conf,
                        fingerprint_score=fp_score,
                        note=f"Fingerprint belongs to {fp_name}",
                    )
                    print("\n❌ ACCESS DENIED - Biometric mismatch\n")
                    session = reset_session()
                    state = set_state(state, PipelineState.IDLE, "biometric mismatch")
                    continue

                decision = fuzzy.make_decision(fp_score, face_conf)
                logger.info(f"🧠 Fuzzy decision: {decision:.3f}")

                if decision >= 0.6:
                    status("GRANTED", face_name, f"Decision {decision:.3f}")
                    record_attendance(
                        name=face_name,
                        status="success",
                        image_path=face_data.get("capture_path"),
                        face_score=face_data.get("score"),
                        face_confidence=face_conf,
                        fingerprint_score=fp_score,
                        decision=decision,
                        note="Access granted",
                    )
                    print(f"\n✅ ACCESS GRANTED\n👋 Welcome {face_name}!\n")
                elif decision >= 0.4:
                    status("REVIEW", face_name, f"Decision {decision:.3f}", "Can xac minh")
                    record_attendance(
                        name=face_name,
                        status="manual_review",
                        image_path=face_data.get("capture_path"),
                        face_score=face_data.get("score"),
                        face_confidence=face_conf,
                        fingerprint_score=fp_score,
                        decision=decision,
                        note="Manual verification needed",
                    )
                    print(f"\n⚠️ UNCERTAIN\nManual verification needed.\n")
                else:
                    status("DENIED", face_name, f"Decision {decision:.3f}", "Do tin cay thap")
                    record_attendance(
                        name=face_name,
                        status="denied",
                        image_path=face_data.get("capture_path"),
                        face_score=face_data.get("score"),
                        face_confidence=face_conf,
                        fingerprint_score=fp_score,
                        decision=decision,
                        note="Low confidence",
                    )
                    print(f"\n❌ ACCESS DENIED\nLow confidence.\n")

                session = reset_session()
                state = set_state(state, PipelineState.IDLE, "verify complete")

    except KeyboardInterrupt:
        logger.info("🛑 Stopping pipeline...")
        status("STOPPING", "Dang dung he thong")

    finally:
        stop_event.set()

        for p in processes:
            p.join(timeout=5)

            if p.is_alive():
                logger.warning(f"{p.name} timeout - terminating.")
                p.terminate()
                p.join()

        if sensor:
            try:
                sensor.clean()
            except Exception as e:
                logger.error(f"Error cleaning sensor: {e}")

        logger.info("✅ Shutdown complete.")
        status("STOPPED", "He thong da dung")


if __name__ == "__main__":
    main()
