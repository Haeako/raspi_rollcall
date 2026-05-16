"""
main_pipeline_sync.py
---------------------

Synchronous 1-1 biometric pipeline with State Machine.

Flow (State Machine):
    IDLE -> WAIT_FACE -> ENROLL -> IDLE
    IDLE -> WAIT_FACE -> WAIT_FP -> VERIFY -> IDLE

Workers:
    FaceWorker         : captures only when requested
    FingerprintWorker  : scans only when requested

IPC:
    Face uses Event + response Queue(maxsize=1)
    Fingerprint uses command Queue(maxsize=1) + response Queue(maxsize=1)
"""

import sys
import time
import json
import logging
import queue
from enum import Enum, auto
from multiprocessing import Process, Queue, Event
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core components with error handling

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

# =============================================================================
# ENUMS
# =============================================================================

class PipelineState(Enum):
    IDLE = auto()
    WAIT_FACE = auto()
    ENROLL = auto()
    WAIT_FP = auto()
    VERIFY = auto()

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s] %(message)s",
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

CONFIG_PATH = get_config_path()

DEFAULT_CONFIG = {
    "mode": "both",      # face | fingerprint | both
    "trigger": "sensor",

    # Sensor
    "sensor_pin": 26,
    "sensor_cooldown": 3.0,

    # Face
    "face_confidence_thresh": 0.5,
    "face_threshold": 0.4,

    # Timeout
    "face_timeout": 5.0,
    "fp_timeout": 10.0,
    "fp_enroll_timeout": 30.0,
    "worker_startup_timeout": 90.0,
    "idle_log_interval": 5.0,

    # Weights (just filenames, paths resolved by core)
    "det_weight": "det_10g.onnx",
    "rec_weight": "w600k_r50.onnx",

    # Qdrant
    "qdrant_host": "qdrant",
    "qdrant_port": 6333,
}

VALID_MODES = {"face", "fingerprint", "both"}
VALID_TRIGGERS = {"sensor"}


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

    cfg["mode"] = str(cfg.get("mode", DEFAULT_CONFIG["mode"])).lower()
    cfg["trigger"] = str(cfg.get("trigger", DEFAULT_CONFIG["trigger"])).lower()

    if cfg["mode"] not in VALID_MODES:
        logger.warning("Invalid mode=%r, falling back to 'both'", cfg["mode"])
        cfg["mode"] = "both"

    if cfg["trigger"] not in VALID_TRIGGERS:
        logger.warning("Invalid trigger=%r, forcing 'sensor'", cfg["trigger"])
        cfg["trigger"] = "sensor"

    return cfg


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def request_fingerprint_scan(request_queue, response_queue, logger):
    """Request fingerprint scan from worker."""
    clear_queue(request_queue)
    clear_queue(response_queue)
    logger.info("Requesting fingerprint scan...")
    request_queue.put({"action": "search"})
    return time.time()


def request_fingerprint_enroll(request_queue, response_queue, logger):
    """Request fingerprint enrollment from worker."""
    clear_queue(request_queue)
    clear_queue(response_queue)
    logger.info("Requesting fingerprint enrollment...")
    request_queue.put({"action": "enroll"})
    return time.time()


def wait_for_response(response_queue, timeout, logger, label):
    """Wait for one worker response with a timeout."""
    start = time.time()
    while time.time() - start <= timeout:
        try:
            return response_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.05)

    logger.warning("%s timeout", label)
    return {
        "success": False,
        "message": f"{label} timeout",
    }


def reset_session():
    """Return clean per-scan state."""
    return {
        "face": None,
        "fingerprint": None,
        "object_detected_at": 0.0,
        "face_requested_at": 0.0,
        "fp_requested_at": 0.0,
    }


def set_state(current_state, next_state, reason=""):
    """Move state machine to the next state and log transitions."""
    if current_state != next_state:
        suffix = f" ({reason})" if reason else ""
        logger.info("STATE %s -> %s%s", current_state.name, next_state.name, suffix)
    return next_state


def enroll_new_person(
    embedding,
    db,
    mode="face",
    fp_request=None,
    fp_response=None,
    cfg=None,
):

    print("\n" + "=" * 50)
    print("UNKNOWN FACE")
    print("=" * 50)

    if input("Enroll? (y/n): ").strip().lower() != "y":
        return

    name = input("Name: ").strip()

    if not name:
        print("Invalid name.")
        return

    fp_position = -1

    if mode == "both" and fp_request is not None and fp_response is not None:
        print("Place finger on sensor to enroll fingerprint...")
        request_fingerprint_enroll(fp_request, fp_response, logger)
        fp_result = wait_for_response(
            fp_response,
            cfg["fp_enroll_timeout"] if cfg else DEFAULT_CONFIG["fp_enroll_timeout"],
            logger,
            "Fingerprint enrollment",
        )

        if fp_result["success"]:
            fp_position = fp_result["position"]
            print(f"Fingerprint stored at slot #{fp_position}")
        else:
            print(f"Fingerprint enroll failed: {fp_result['message']}")

            if input("Save face only? (y/n): ").strip().lower() != "y":
                return

    db.add(
        embedding,
        name,
        fingerprint_position=fp_position
    )

    print(
        f"Enroll success: "
        f"{name} (fp_slot={fp_position})"
    )


# =============================================================================
# FACE WORKER
# =============================================================================

def face_worker(request_event, response_queue, stop_event, ready_event, cfg):
    """Face detection and recognition worker process."""
    logger.info("Face worker started.")

    try:
        camera = PiCamera()
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
            # Wait for request
            if not request_event.wait(timeout=0.1):
                continue

            request_event.clear()

            try:
                frame = camera.capture_array()
                embs, bbs, confs = model.get_embeding_vector(
                    frame,
                    image_format="BGR",
                )

                if len(embs) == 0:
                    response_queue.put({
                        "success": False,
                        "message": "No face detected",
                    })
                    continue

                # Get best match
                best_idx = int(confs.argmax()) if hasattr(confs, "argmax") else 0
                best_emb = embs[best_idx]
                best_conf = float(confs[best_idx])

                # Search database
                results = db.search(best_emb, top_k=1, threshold=threshold)

                if results:
                    name, score, fp_pos = results[0]
                else:
                    name, score, fp_pos = "Unknown", 0.0, -1

                response_queue.put({
                    "success": True,
                    "name": name,
                    "score": score,
                    "confidence": best_conf,
                    "fp_pos": fp_pos,
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


# =============================================================================
# FINGERPRINT WORKER
# =============================================================================

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
                    result = sensor.enroll(timeout=cfg["fp_enroll_timeout"])
                else:
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main pipeline orchestrator."""
    # Use spawn to ensure clean child processes
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    cfg = load_config()
    mode = "both"
    logger.info("Pipeline initializing...")
    logger.info("Flow: HW-201 -> face -> fingerprint -> fuzzy -> result")

    # -------------------------------------------------------------------------
    # Trigger sensor initialization
    # -------------------------------------------------------------------------

    trigger = cfg["trigger"]
    sensor = None

    if trigger == "sensor":
        try:
            sensor = HW_201_HAL(sensor_pin=cfg["sensor_pin"])
            logger.info("HW-201 sensor ready.")
        except Exception as e:
            logger.error(f"HW-201 initialization failed: {e}")
            return

    # -------------------------------------------------------------------------
    # Initialize IPC (Inter-Process Communication)
    # -------------------------------------------------------------------------

    stop_event = Event()

    face_request = Event()
    face_response = Queue(maxsize=1)
    face_ready = Event()

    fp_request = Queue(maxsize=1)
    fp_response = Queue(maxsize=1)
    fp_ready = Event()

    # -------------------------------------------------------------------------
    # Initialize database and models
    # -------------------------------------------------------------------------

    try:
        db = Qdrant_db(
            host=cfg["qdrant_host"],
            port=cfg["qdrant_port"],
        )
        logger.info("Qdrant database connected.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return

    fuzzy = None
    try:
        fuzzy = FuzzyModel()
        logger.info("Fuzzy model initialized.")
    except Exception as e:
        logger.error(f"Fuzzy model initialization failed: {e}")
        return

    # Ensure captures directory exists
    try:
        ensure_capture_dir()
    except Exception as e:
        logger.warning(f"Could not create captures directory: {e}")

    # -------------------------------------------------------------------------
    # Start worker processes
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Main state machine loop
    # -------------------------------------------------------------------------

    cooldown = cfg["sensor_cooldown"]
    last_trigger = 0

    logger.info("✅ Pipeline running. Press Ctrl+C to stop.")

    state = PipelineState.IDLE
    session = reset_session()
    last_idle_log = 0.0

    try:
        while True:
            if state == PipelineState.IDLE:
                # ==============================================================
                # TRIGGER - HW-201 object detection only
                # ==============================================================
                if not sensor.detect():
                    now = time.time()
                    if now - last_idle_log >= cfg["idle_log_interval"]:
                        logger.info("STATE IDLE: waiting for HW-201 trigger on GPIO %s", cfg["sensor_pin"])
                        last_idle_log = now
                    time.sleep(0.05)
                    continue

                now = time.time()
                if now - last_trigger < cooldown:
                    time.sleep(0.05)
                    continue

                last_trigger = now
                logger.info("📍 Sensor triggered")
                session = reset_session()
                session["object_detected_at"] = now
                session["face_requested_at"] = request_face_capture(face_request, face_response, logger)
                state = set_state(state, PipelineState.WAIT_FACE, "sensor trigger")

            elif state == PipelineState.WAIT_FACE:
                # ==============================================================
                # WAIT FOR FACE RESULT
                # ==============================================================
                try:
                    face_data = face_response.get_nowait()
                    
                    if not face_data["success"]:
                        logger.warning(f"⚠️ Face detection failed: {face_data['message']}")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "face failed")
                        continue

                    session["face"] = face_data
                    face_name = face_data["name"]
                    logger.info(f"👤 Face detected: {face_name}")

                    if face_name == "Unknown":
                        state = set_state(state, PipelineState.ENROLL, "unknown face")
                    else:
                        logger.info(f"👋 Hello {face_name}")
                        session["fp_requested_at"] = request_fingerprint_scan(fp_request, fp_response, logger)
                        state = set_state(state, PipelineState.WAIT_FP, "face matched")

                except queue.Empty:
                    if time.time() - session["face_requested_at"] > cfg["face_timeout"]:
                        logger.warning("⏱️ Face detection timeout")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "face timeout")
                    else:
                        time.sleep(0.05)

            elif state == PipelineState.ENROLL:
                # ==============================================================
                # ENROLL NEW PERSON
                # ==============================================================
                enroll_new_person(
                    session["face"]["embedding"],
                    db,
                    mode=mode,
                    fp_request=fp_request,
                    fp_response=fp_response,
                    cfg=cfg,
                )
                session = reset_session()
                state = set_state(state, PipelineState.IDLE, "enroll complete")

            elif state == PipelineState.WAIT_FP:
                # ==============================================================
                # WAIT FOR FINGERPRINT RESULT
                # ==============================================================
                try:
                    fp_data = fp_response.get_nowait()
                    
                    if not fp_data["success"]:
                        logger.warning(f"⚠️ Fingerprint failed: {fp_data['message']}")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "fingerprint failed")
                        continue

                    session["fingerprint"] = fp_data
                    state = set_state(state, PipelineState.VERIFY, "fingerprint matched")

                except queue.Empty:
                    if time.time() - session["fp_requested_at"] > cfg["fp_timeout"]:
                        logger.warning("⏱️ Fingerprint scan timeout")
                        session = reset_session()
                        state = set_state(state, PipelineState.IDLE, "fingerprint timeout")
                    else:
                        time.sleep(0.05)

            elif state == PipelineState.VERIFY:
                # ==============================================================
                # VERIFY - Cross-check face and fingerprint
                # ==============================================================
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
                    print("\n❌ ACCESS DENIED - Biometric mismatch\n")
                    session = reset_session()
                    state = set_state(state, PipelineState.IDLE, "biometric mismatch")
                    continue

                # Make fuzzy decision
                decision = fuzzy.make_decision(fp_score, face_conf)
                logger.info(f"🧠 Fuzzy decision: {decision:.3f}")

                if decision >= 0.6:
                    print(f"\n✅ ACCESS GRANTED\n👋 Welcome {face_name}!\n")
                elif decision >= 0.4:
                    print(f"\n⚠️ UNCERTAIN\nManual verification needed.\n")
                else:
                    print(f"\n❌ ACCESS DENIED\nLow confidence.\n")

                session = reset_session()
                state = set_state(state, PipelineState.IDLE, "verify complete")

    except KeyboardInterrupt:
        logger.info("🛑 Stopping pipeline...")

    finally:
        # Clean shutdown
        stop_event.set()

        # Wait for workers to finish
        for p in processes:
            p.join(timeout=5)

            if p.is_alive():
                logger.warning(f"{p.name} timeout - terminating.")
                p.terminate()
                p.join()

        # Clean up hardware
        if sensor:
            try:
                sensor.clean()
            except Exception as e:
                logger.error(f"Error cleaning sensor: {e}")

        logger.info("✅ Shutdown complete.")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()
