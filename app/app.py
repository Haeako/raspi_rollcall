import sys
sys.path.append("core/src")
from AS608 import AS608_HAL
from HW_201 import HW_201_HAL
from Pi_cam import PiCamera
def main():
    sensor = HW_201_HAL(sensor_pin=26)
    camera = PiCamera()

    cam_active = False
    cam_start_time = 0
    CAM_DURATION = 5  # giây

    print("🔄 Bắt đầu vòng lặp detect...")

    try:
        while True:
            detected = sensor.detect()

            # Có vật → bật camera nếu chưa bật
            if detected and not cam_active:
                print("⚠️  Phát hiện vật cản!")
                camera.start()
                camera.capture(f"capture_{int(time.time())}.jpg")
                cam_active = True
                cam_start_time = time.time()

            # Camera đang bật → kiểm tra hết 5 giây chưa
            if cam_active:
                elapsed = time.time() - cam_start_time
                if elapsed >= CAM_DURATION:
                    camera.stop()
                    cam_active = False
                    print("⏹️  Hết 5 giây, tắt camera")

            time.sleep(0.2)  # poll mỗi 200ms

    except KeyboardInterrupt:
        print("\n🛑 Dừng chương trình")
    finally:
        if cam_active:
            camera.stop()
        camera.close()
        sensor.cleanup()


if __name__ == "__main__":
    main()