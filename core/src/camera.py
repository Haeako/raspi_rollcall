"""Picamera2 wrapper that returns RGB frames."""

import logging
import time
from pathlib import Path

import numpy as np
from picamera2 import Picamera2

logger = logging.getLogger(__name__)


class PiCamera:
    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        framerate: int = 10,
        warmup_sec: float = 1.0,
        buffer_count: int = 2,
    ) -> None:
        self.width = width
        self.height = height
        self.framerate = framerate
        self.warmup_sec = warmup_sec
        self.buffer_count = buffer_count
        self.picam = Picamera2()
        self._configure()
        self._start()

    def _configure(self) -> None:
        config = self.picam.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            },
            controls={"FrameRate": self.framerate},
            buffer_count=self.buffer_count,
        )
        self.picam.configure(config)

    def _start(self) -> None:
        last_error = None
        for attempt in range(3):
            try:
                self.picam.start()
                time.sleep(self.warmup_sec)
                logger.info(
                    "Camera started (%sx%s@%s, buffers=%s)",
                    self.width,
                    self.height,
                    self.framerate,
                    self.buffer_count,
                )
                return
            except Exception as error:
                last_error = error
                logger.warning("Camera start failed (%s/3): %s", attempt + 1, error)
                self._stop_quietly()
                time.sleep(0.5)

        raise RuntimeError(f"Camera start failed after retries: {last_error}")

    def _stop_quietly(self) -> None:
        try:
            self.picam.stop()
        except Exception:
            pass

    def capture_array(self) -> np.ndarray:
        try:
            return self.picam.capture_array("main")
        except Exception as error:
            logger.warning("Camera capture failed; restarting camera: %s", error)
            self._stop_quietly()
            time.sleep(0.5)
            self._start()
            return self.picam.capture_array("main")

    def capture(self, filepath: str = "capture.jpg") -> str:
        import cv2

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        frame = cv2.cvtColor(self.capture_array(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, frame)
        return filepath

    def close(self) -> None:
        self._stop_quietly()
        self.picam.close()


def main() -> None:
    cam = PiCamera()
    try:
        path = cam.capture("captures/test.jpg")
        print(f"Captured: {path}")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
