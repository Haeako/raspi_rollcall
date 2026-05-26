import time


class HW_201_HAL:
    """
    HW-201 obstacle sensor driver using gpiozero + lgpio only.

    Raspberry Pi 5 requires the modern /dev/gpiochip interface. gpiozero with
    the lgpio pin factory talks to that interface directly.
    """

    def __init__(self, sensor_pin: int = 26, chip: int = 0):
        self.sensor_pin = sensor_pin
        self.chip = chip
        self.backend = "gpiozero-lgpio"
        self.device = None
        self.pin_factory = None

        try:
            from gpiozero import DigitalInputDevice
            from gpiozero.pins.lgpio import LGPIOFactory

            self.pin_factory = LGPIOFactory(chip=self.chip)
            self.device = DigitalInputDevice(
                self.sensor_pin,
                pull_up=None,
                active_state=False,
                pin_factory=self.pin_factory,
            )
            print(f"[HW-201] GPIO ready via {self.backend} on BCM {self.sensor_pin}")
        except Exception as error:
            self.clean()
            raise RuntimeError(
                "Cannot initialize HW-201 GPIO with gpiozero-lgpio. "
                "Install python3-gpiozero and python3-lgpio, expose "
                "/dev/gpiochip* when running in Docker, and make sure no other "
                f"process is using BCM GPIO {self.sensor_pin}. "
                f"gpiozero-lgpio error: {error}"
            ) from error

    def detect(self) -> bool:
        """LOW = co vat can -> True."""
        return bool(self.device and self.device.is_active)

    def clean(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None

        if self.pin_factory is not None:
            self.pin_factory.close()
            self.pin_factory = None


def main() -> None:
    sensor = HW_201_HAL(26)
    try:
        while True:
            print("Detected:", sensor.detect())
            time.sleep(0.5)
    finally:
        sensor.clean()


if __name__ == "__main__":
    main()
