"""
main_pipeline_sync.py
---------------------

Kiến trúc synchronous 1-1 biometric pipeline với State Machine.

Flow (State Machine):
    IDLE -> WAIT_FACE -> ENROLL -> WAIT_FP -> VERIFY -> IDLE

Workers:
    FaceWorker         : chỉ capture khi được request
    FingerprintWorker  : chỉ scan khi được request

IPC:
    Event + Queue(maxsize=1)
"""

import os
import sys
import time
import json
import logging
import queue
import multiprocessing

from enum import Enum, auto
from multiprocessing import Process, Queue, Event

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../")
    )
)

from core import (
    PiCamera,
    Qdrant_db,
    FaceModel,
    FuzzyModel,
    AS608_HAL,
    HW_201_HAL,
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

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config.json"
)

DEFAULT_CONFIG = {
    "trigger": "sensor",  # sensor | terminal | auto

    # Sensor
    "sensor_pin": 26,
    "sensor_cooldown": 3.0,

    # Face
    "face_confidence_thresh": 0.5,
    "face_threshold": 0.4,

    # Timeout
    "face_timeout": 5.0,
    "fp_timeout": 10.0,

    # Weights
    "det_weight": "../weights/det_10g.onnx",
    "rec_weight": "../weights/w600k_r50.onnx",

    # Qdrant
    "qdrant_host": "qdrant",
    "qdrant_port": 6333,
}


def load_config(path=CONFIG_PATH):

    cfg = DEFAULT_CONFIG.copy()

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))

            logger.info(f"Config loaded: {path}")

        except Exception as e:
            logger.warning(f"Load config failed: {e}")

    return cfg


# =============================================================================
# ENROLL
# =============================================================================

def enroll_new_person(
    embedding,
    db,
    fp_sensor=None
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

    if fp_sensor is not None:

        print("Place finger on sensor...")

        fp_result = fp_sensor.enroll()

        if fp_result.success:

            fp_position = fp_result.position

            print(
                f"Fingerprint stored at slot #{fp_position}"
            )

        else:

            print(
                f"Fingerprint enroll failed: "
                f"{fp_result.message}"
            )

            if input(
                "Save face only? (y/n): "
            ).strip().lower() != "y":
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

def face_worker(
    request_event,
    response_queue,
    stop_event,
    cfg
):

    logger.info("Face worker started.")

    camera = PiCamera()

    model = FaceModel(
        det_weight=cfg["det_weight"],
        rec_weight=cfg["rec_weight"],
        confidence_thresh=cfg["face_confidence_thresh"],
    )

    db = Qdrant_db(
        host=cfg["qdrant_host"],
        port=cfg["qdrant_port"],
    )

    threshold = cfg["face_threshold"]

    try:

        while not stop_event.is_set():

            # wait request
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

                best_idx = (
                    int(confs.argmax())
                    if hasattr(confs, "argmax")
                    else 0
                )

                best_emb = embs[best_idx]

                best_conf = float(
                    confs[best_idx]
                )

                results = db.search(
                    best_emb,
                    top_k=1,
                    threshold=threshold,
                )

                name, score, fp_pos = results[0]

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

                logger.error(f"Face error: {e}")

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

def fingerprint_worker(
    request_event,
    response_queue,
    stop_event,
    cfg,
):

    logger.info("Fingerprint worker started.")

    try:
        sensor = AS608_HAL()

    except RuntimeError as e:

        logger.error(f"AS608 init failed: {e}")
        return

    try:

        while not stop_event.is_set():

            if not request_event.wait(timeout=0.1):
                continue

            request_event.clear()

            try:

                result = sensor.search(timeout=cfg["fp_timeout"])

                if result.success:

                    response_queue.put({
                        "success": True,
                        "position": result.position,
                        "score": result.score,
                        "ts": time.time(),
                    })

                else:

                    response_queue.put({
                        "success": False,
                        "message": result.message,
                    })

            except Exception as e:

                logger.error(f"Fingerprint error: {e}")

                response_queue.put({
                    "success": False,
                    "message": str(e),
                })

    finally:

        logger.info("Fingerprint worker stopped.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    multiprocessing.set_start_method(
        "spawn",
        force=True
    )

    cfg = load_config()

    # -------------------------------------------------------------------------
    # Trigger sensor
    # -------------------------------------------------------------------------

    trigger = cfg["trigger"]

    sensor = None

    if trigger == "sensor":

        try:

            sensor = HW_201_HAL(
                sensor_pin=cfg["sensor_pin"]
            )

            logger.info("HW-201 ready.")

        except Exception as e:

            logger.warning(
                f"HW-201 failed: {e}"
            )

            trigger = "terminal"

    # -------------------------------------------------------------------------
    # IPC
    # -------------------------------------------------------------------------

    stop_event = Event()

    face_request = Event()
    face_response = Queue(maxsize=1)

    fp_request = Event()
    fp_response = Queue(maxsize=1)

    # -------------------------------------------------------------------------
    # DB
    # -------------------------------------------------------------------------

    db = Qdrant_db(
        host=cfg["qdrant_host"],
        port=cfg["qdrant_port"],
    )

    fuzzy = FuzzyModel()

    try:
        fp_sensor_main = AS608_HAL()

    except RuntimeError:

        fp_sensor_main = None

    # -------------------------------------------------------------------------
    # Workers
    # -------------------------------------------------------------------------

    processes = [

        Process(
            target=face_worker,
            args=(
                face_request,
                face_response,
                stop_event,
                cfg,
            ),
            daemon=True,
            name="FaceWorker",
        ),

        Process(
            target=fingerprint_worker,
            args=(
                fp_request,
                fp_response,
                stop_event,
                cfg,
            ),
            daemon=True,
            name="FingerprintWorker",
        ),
    ]

    for p in processes:

        p.start()

        logger.info(
            f"Started {p.name} "
            f"(PID={p.pid})"
        )

    # -------------------------------------------------------------------------
    # Main loop (State Machine)
    # -------------------------------------------------------------------------

    cooldown = cfg["sensor_cooldown"]
    last_trigger = 0

    logger.info("Pipeline running.")

    state = PipelineState.IDLE
    current_face_data = None
    current_fp_data = None
    face_request_time = 0
    fp_request_time = 0

    try:

        while True:

            if state == PipelineState.IDLE:
                
                # ==============================================================
                # TRIGGER
                # ==============================================================
                if trigger == "terminal":
                    cmd = input("\n[Enter] Scan | [q] Quit: ").strip().lower()
                    if cmd == "q":
                        break
                    
                    while not face_response.empty():
                        try: face_response.get_nowait()
                        except: pass
                    
                    logger.info("Capturing face...")
                    face_request.set()
                    face_request_time = time.time()
                    state = PipelineState.WAIT_FACE

                elif trigger == "sensor":
                    if not sensor.detect():
                        time.sleep(0.05)
                        continue
                    now = time.time()
                    if now - last_trigger < cooldown:
                        time.sleep(0.05)
                        continue
                    last_trigger = now
                    
                    logger.info("Sensor triggered.")
                    while not face_response.empty():
                        try: face_response.get_nowait()
                        except: pass
                    
                    logger.info("Capturing face...")
                    face_request.set()
                    face_request_time = time.time()
                    state = PipelineState.WAIT_FACE

                elif trigger == "auto":
                    time.sleep(1)
                    while not face_response.empty():
                        try: face_response.get_nowait()
                        except: pass
                    
                    logger.info("Capturing face...")
                    face_request.set()
                    face_request_time = time.time()
                    state = PipelineState.WAIT_FACE


            elif state == PipelineState.WAIT_FACE:
                
                # ==============================================================
                # FACE REQUEST
                # ==============================================================
                try:
                    face_data = face_response.get_nowait()
                    
                    if not face_data["success"]:
                        logger.warning(f"Face failed: {face_data['message']}")
                        state = PipelineState.IDLE
                        continue

                    current_face_data = face_data
                    face_name = face_data["name"]
                    logger.info(f"Face result: {face_name}")

                    if face_name == "Unknown":
                        state = PipelineState.ENROLL
                    else:
                        logger.info(f"Hello {face_name}")
                        while not fp_response.empty():
                            try: fp_response.get_nowait()
                            except: pass

                        logger.info("Waiting fingerprint...")
                        fp_request.set()
                        fp_request_time = time.time()
                        state = PipelineState.WAIT_FP

                except queue.Empty:
                    if time.time() - face_request_time > cfg["face_timeout"]:
                        logger.warning("Face timeout.")
                        state = PipelineState.IDLE
                    else:
                        time.sleep(0.05)


            elif state == PipelineState.ENROLL:
                
                # ==============================================================
                # UNKNOWN → ENROLL
                # ==============================================================
                enroll_new_person(
                    current_face_data["embedding"],
                    db,
                    fp_sensor_main,
                )
                state = PipelineState.IDLE


            elif state == PipelineState.WAIT_FP:
                
                # ==============================================================
                # FINGERPRINT REQUEST
                # ==============================================================
                try:
                    fp_data = fp_response.get_nowait()
                    
                    if not fp_data["success"]:
                        logger.warning(f"Fingerprint failed: {fp_data['message']}")
                        state = PipelineState.IDLE
                        continue

                    current_fp_data = fp_data
                    state = PipelineState.VERIFY

                except queue.Empty:
                    if time.time() - fp_request_time > cfg["fp_timeout"]:
                        logger.warning("Fingerprint timeout.")
                        state = PipelineState.IDLE
                    else:
                        time.sleep(0.05)


            elif state == PipelineState.VERIFY:
                
                # ==============================================================
                # CROSS CHECK & VALIDATE & FUZZY
                # ==============================================================
                face_name = current_face_data["name"]
                fp_pos = current_fp_data["position"]
                fp_name = db.get_name_by_fp_position(fp_pos)
                score = current_fp_data["score"]
                conf = current_face_data["confidence"]

                logger.info(f"FP result: {fp_name}")

                if fp_name != face_name:
                    logger.warning(f"MISMATCH (face={face_name}, fp={fp_name})")
                    print("\n❌ ACCESS DENIED")
                    print("Biometric mismatch.\n")
                    state = PipelineState.IDLE
                    continue

                decision = fuzzy.make_decision(score, conf)
                logger.info(f"Fuzzy={decision:.3f}")

                if decision >= 0.6:
                    print("\n✅ ACCESS GRANTED")
                    print(f"Welcome {face_name}")
                    print(f"FP score: {score}")
                    print(f"Face conf: {conf:.3f}\n")
                elif decision >= 0.4:
                    print("\n⚠️ UNCERTAIN")
                    print("Need manual verification.\n")
                else:
                    print("\n❌ ACCESS DENIED")
                    print("Low confidence.\n")

                state = PipelineState.IDLE

    except KeyboardInterrupt:

        logger.info("Stopping pipeline...")

    finally:

        stop_event.set()

        for p in processes:

            p.join(timeout=5)

            if p.is_alive():

                logger.warning(
                    f"{p.name} timeout terminate."
                )

                p.terminate()

                p.join()

        if sensor:
            sensor.clean()

        logger.info("Shutdown complete.")


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    main()