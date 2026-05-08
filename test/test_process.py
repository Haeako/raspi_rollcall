# test/fuzzy.py

from multiprocessing import Process, Queue
from core import FaceModel, PiCamera, FuzzyModel

import random
import time


def face_worker(face_queue):

    camera = PiCamera()

    model = FaceModel()

    while True:

        try:

            frame = camera.capture_array()
            print("[DEBUG]: capture frame")
            emb, bbs, confs = model.get_embeding_vector(frame)
            print(f"[DEBUG]: GOT {bbs} face")
            if len(bbs) > 0:

                confidence = float(max(confs))

                face_queue.put({
                    "confidence": confidence
                })

            time.sleep(0.1)

        except Exception as e:

            print("[FACE ERROR]", e)

            time.sleep(1)


def fingerprint_worker(fp_queue):
    
    while True:

        try:

            time.sleep(2)

            score = random.randint(0, 300)
            print(f"[DEBUG]: generate match socre {score}")
            fp_queue.put({
                "score": score
            })

        except Exception as e:

            print("[FP ERROR]", e)

            time.sleep(1)


def fuzzy_worker(face_queue, fp_queue):

    fuzzy = FuzzyModel()

    latest_conf = None
    latest_score = None

    while True:

        try:

            while not face_queue.empty():

                data = face_queue.get()

                latest_conf = data["confidence"]

            while not fp_queue.empty():

                data = fp_queue.get()

                latest_score = data["score"]

            if (
                latest_conf is not None and
                latest_score is not None
            ):

                decision = fuzzy.make_decision(
                    latest_score,
                    latest_conf
                )

                print({
                    "score": latest_score,
                    "confidence": latest_conf,
                    "decision": decision
                })

                latest_conf = None
                latest_score = None

            time.sleep(0.05)

        except Exception as e:

            print("[FUZZY ERROR]", e)

            time.sleep(1)


if __name__ == "__main__":

    face_queue = Queue()

    fp_queue = Queue()

    p1 = Process(
        target=face_worker,
        args=(face_queue,)
    )

    p2 = Process(
        target=fingerprint_worker,
        args=(fp_queue,)
    )

    p3 = Process(
        target=fuzzy_worker,
        args=(face_queue, fp_queue)
    )

    p1.start()
    p2.start()
    p3.start()

    try:

        p1.join()
        p2.join()
        p3.join()

    except KeyboardInterrupt:

        print("\n[INFO] Stopping processes...")

        p1.terminate()
        p2.terminate()
        p3.terminate()

        p1.join()
        p2.join()
        p3.join()

        print("[INFO] Shutdown complete")