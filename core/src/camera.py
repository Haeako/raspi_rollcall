from pathlib import Path
import time
import numpy as np
from picamera2 import Picamera2


class PiCamera:
    """
    Wrapper picamera2 tối ưu cho inference realtime.

    Thay đổi so với bản gốc:
      - create_video_configuration thay vì create_still_configuration
        → không còn bị block 1-3s mỗi lần capture_array
      - Resolution mặc định 640x480 (đủ cho SCRFD, nhẹ hơn 16x so với 2592x1944)
      - Format BGR888 → thẳng vào OpenCV/SCRFD, không cần convert
      - Warm-up tự động sau start để AE/AWB ổn định
      - capture() vẫn dùng được để lưu file jpg
    """

    def __init__(
        self,
        width:      int   = 640,
        height:     int   = 480,
        framerate:  int   = 30,
        warmup_sec: float = 2.0,
    ):
        self.picam = Picamera2()

        # Video config: non-blocking, stream liên tục
        config = self.picam.create_video_configuration(
            main={
                "size":   (width, height),
                "format": "BGR888",   # BGR → dùng thẳng cho OpenCV & SCRFD
            },
            controls={
                "FrameRate": framerate,
            },
            buffer_count=4,           # buffer nhỏ → latency thấp
        )

        self.picam.configure(config)
        self.picam.start()

        # Warm-up: chờ AE/AWB hội tụ
        time.sleep(warmup_sec)
        print("[INFO] CAMERA START")

    def capture_array(self) -> np.ndarray:
        """
        Trả về frame hiện tại dạng numpy array BGR uint8.
        Non-blocking, lấy frame mới nhất từ buffer (~1ms).
        """
        return self.picam.capture_array("main")

    def capture(self, filepath: str = "capture.jpg") -> str:
        """
        Lưu frame hiện tại ra file jpg.
        Dùng capture_array + imencode để tránh switch sang still mode.
        """
        import cv2

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        frame = self.capture_array()                  # BGR
        cv2.imwrite(filepath, frame)
        return filepath

    def close(self):
        self.picam.stop()
        self.picam.close()


if __name__ == "__main__":
    cam = PiCamera()
    path = cam.capture("captures/test.jpg")
    print(f"Đã chụp: {path}")
    cam.close()