import RPi.GPIO as GPIO
import time


class HW_201_HAL:
    def __init__(self, sensor_pin: int = 26):
        self.sensor_pin = sensor_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN)

    def detect(self) -> bool:
        """LOW = có vật cản → True."""
        return GPIO.input(self.sensor_pin) == 0

    def clean(self):           # fix: phải là instance method (có self)
        GPIO.cleanup()


def main():
    sensor = HW_201_HAL(26)
    try:
        while True:
            print("Detected:", sensor.detect())
            time.sleep(0.5)
    finally:
        sensor.clean()


if __name__ == "__main__":
    main()