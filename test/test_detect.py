from core import FaceModel, HW_201_HAL, PiCamera
import time
import cv2

sensor = HW_201_HAL(26)

camera = PiCamera()

model = FaceModel()

COOLDOWN = 5

last_trigger = 0

while True:

    now = time.time()

    if sensor.detect() and (now - last_trigger > COOLDOWN):

        print("[INFO] Motion detected")

        last_trigger = now

        start_time = time.time()

        face_found = False

        while time.time() - start_time < COOLDOWN:

            frame = camera.capture_array()

            bbs, ccs, kpss = model.detect(frame)

            if len(bbs) > 0:

                filename = (
                    f"/workspace/raspi_rollcall/captures/"
                    f"{int(time.time())}.jpg"
                )

                cv2.imwrite(filename, frame)

                print(f"[INFO] Face saved: {filename}")

                face_found = True

                break

            time.sleep(0.1)

        if not face_found:

            print("[INFO] No face detected")

    time.sleep(0.05)