from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()

frame = picam2.capture_array()

import cv2
cv2.imwrite("test.jpg", frame)

picam2.stop()
