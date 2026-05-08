import RPi.GPIO as GPIO
import time

class HW_201_HAL:
    def __init__(self, sensor_pin):
        self.sensor_pin = sensor_pin  # GPIO17
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.sensor_pin, GPIO.IN)

    def detect(self):
        value = GPIO.input(self.sensor_pin)
        if value == 0:
            return True
        else:
            return False
    def clean():
        GPIO.cleanup()

def main():
    sensor = HW_201_HAL(26)
    while True:
        print(sensor.detect())
        time.sleep(2)
if __name__ == "__main__":
    main()