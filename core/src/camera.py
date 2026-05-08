from pathlib import Path
import time
from picamera2 import Picamera2


class PiCamera:
    def __init__(self):
        self.picam = Picamera2()

        config = self.picam.create_still_configuration()

        self.picam.configure(config)

        self.picam.start()
        print("[INFO] CAMERA START")
        time.sleep(2)

    def capture(self, filepath: str = "capture.jpg") -> str:

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.picam.capture_file(filepath)

        return filepath

    def capture_array(self):

        return self.picam.capture_array()

    def close(self):
        self.picam.close()


if __name__ == "__main__":

    cam = PiCamera()

    path = cam.capture(
        "captures/test.jpg"
    )

    print(f"Đã chụp: {path}")

    cam.close()