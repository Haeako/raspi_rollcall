import time


class HW_201_HAL:
    def __init__(self, sensor_pin: int = 26):
        self.sensor_pin = sensor_pin
        self.backend = None
        self.device = None
        self.gpio = None

        try:
            from gpiozero import DigitalInputDevice
            from gpiozero.pins.lgpio import LGPIOFactory

            self.device = DigitalInputDevice(
                self.sensor_pin,
                pull_up=None,
                active_state=False,
                pin_factory=LGPIOFactory(),
            )
            self.backend = "gpiozero-lgpio"
            print(f"[HW-201] GPIO ready via {self.backend} on BCM {self.sensor_pin}")
            return
        except Exception as gpiozero_error:
            try:
                import RPi.GPIO as GPIO

                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.sensor_pin, GPIO.IN)
                self.gpio = GPIO
                self.backend = "RPi.GPIO"
                print(f"[HW-201] GPIO ready via {self.backend} on BCM {self.sensor_pin}")
                return
            except Exception as rpi_gpio_error:
                raise RuntimeError(
                    "Cannot initialize HW-201 GPIO. "
                    "On Raspberry Pi 5 or Docker, install/use gpiozero + lgpio "
                    "and expose /dev/gpiochip*. "
                    f"gpiozero-lgpio error: {gpiozero_error}; "
                    f"RPi.GPIO error: {rpi_gpio_error}"
                ) from rpi_gpio_error

    def detect(self) -> bool:
        """LOW = có vật cản → True."""
        if self.backend == "gpiozero-lgpio":
            return bool(self.device.is_active)
        return self.gpio.input(self.sensor_pin) == 0

    def clean(self):           # fix: phải là instance method (có self)
        if self.backend == "gpiozero-lgpio" and self.device is not None:
            self.device.close()
        elif self.gpio is not None:
            self.gpio.cleanup()


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
