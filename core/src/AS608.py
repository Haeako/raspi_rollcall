"""AS608/R307 fingerprint sensor driver over UART."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class FingerprintResult:
    success: bool
    message: str
    score: int = 0
    position: int = -1


class AS608_HAL:
    """Hardware abstraction layer for the AS608 fingerprint sensor."""

    UART_PORT = "/dev/ttyAMA0"
    BAUD_RATE = 57600
    ADDRESS = 0xFFFFFFFF
    PASSWORD = 0x00000000

    def __init__( self, port: str = UART_PORT, baud: int = BAUD_RATE,
        address: int = ADDRESS, password: int = PASSWORD, ) -> None:
        
        from pyfingerprint.pyfingerprint import PyFingerprint

        try:
            self.sensor = PyFingerprint(port, baud, address, password)
        except Exception as error:
            raise RuntimeError(f"Không thể mở cổng UART '{port}': {error}") from error

        if not self.sensor.verifyPassword():
            raise RuntimeError("FAIL to init sensor")

        count = self.sensor.getTemplateCount()
        capacity = self.sensor.getStorageCapacity()
        print(f"[AS608] OK - Templates: {count}/{capacity}")

    def enroll(self, timeout: float = 0.0) -> FingerprintResult:
        """Enroll a new fingerprint with two scans."""
        try:
            deadline = time.time() + timeout if timeout > 0 else None

            first_scan = self._wait_for_image(
                self._remaining_timeout(deadline),
                "Timeout khi chờ vân tay lần 1.",
            )
            if not first_scan.success:
                return first_scan

            self.sensor.convertImage(0x01)

            position, _ = self.sensor.searchTemplate()
            if position >= 0:
                return FingerprintResult(
                    success=False,
                    message=f"Vân tay đã tồn tại tại vị trí #{position}.",
                    position=position,
                )

            print("[AS608] Nhấc ngón tay... Đặt lại sau 2 giây.")
            time.sleep(2)

            second_scan = self._wait_for_image(
                self._remaining_timeout(deadline),
                "Timeout khi chờ vân tay lần 2.",
            )
            if not second_scan.success:
                return second_scan

            self.sensor.convertImage(0x02)
            if self.sensor.compareCharacteristics() == 0:
                return FingerprintResult(
                    success=False,
                    message="Hai lần quét không khớp nhau.",
                )

            self.sensor.createTemplate()
            position = self.sensor.storeTemplate()
            return FingerprintResult(
                success=True,
                message=f"Đăng ký thành công tại vị trí #{position}.",
                position=position,
            )
        except Exception as error:
            return FingerprintResult(False, f"Enroll thất bại: {error}")

    def search(self, timeout: float = 0.0) -> FingerprintResult:
        """Read a fingerprint and search the sensor template database."""
        try:
            scan = self._wait_for_image(timeout, "Timeout: Không thấy vân tay.")
            if not scan.success:
                return scan

            self.sensor.convertImage(0x01)
            position, score = self.sensor.searchTemplate()
            if position == -1:
                return FingerprintResult(
                    success=False,
                    message="Không tìm thấy vân tay khớp.",
                )

            return FingerprintResult(
                success=True,
                message=f"Khớp tại vị trí #{position} với độ chính xác {score}.",
                score=int(score),
                position=int(position),
            )
        except Exception as error:
            return FingerprintResult(False, f"Search thất bại: {error}")

    def delete(self, position: int) -> FingerprintResult:
        """Delete one fingerprint template."""
        try:
            if self.sensor.deleteTemplate(position):
                return FingerprintResult(
                    success=True,
                    message=f"Đã xoá template tại vị trí #{position}.",
                    position=position,
                )
            return FingerprintResult(
                success=False,
                message=f"Không thể xoá template tại vị trí #{position}.",
            )
        except Exception as error:
            return FingerprintResult(False, f"Delete thất bại: {error}")

    def empty(self) -> FingerprintResult:
        """Delete all fingerprint templates."""
        try:
            if self.sensor.clearDatabase():
                return FingerprintResult(True, "[AS608]ALLCLEAR")
            return FingerprintResult(False, "[AS608]FAILED_TO_CLEAR_DATABASE")
        except Exception as error:
            return FingerprintResult(False, f"{error}")

    def count(self) -> int:
        return self.sensor.getTemplateCount()

    def _remaining_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return 0.0
        return max(0.01, deadline - time.time())

    def _wait_for_image(self, timeout: float, timeout_message: str) -> FingerprintResult:
        start_time = time.time()
        while not self.sensor.readImage():
            if timeout > 0 and time.time() - start_time > timeout:
                return FingerprintResult(False, timeout_message)
            time.sleep(0.05)
        return FingerprintResult(True, "Image captured.")


def main() -> None:
    try:
        hal = AS608_HAL()
    except RuntimeError as error:
        print(f"[ERROR] Không khởi tạo được sensor: {error}")
        return

    print("=== Enroll ===")
    result = hal.enroll()
    print(result.message)

    time.sleep(2)

    print("=== Search ===")
    result = hal.search()
    print(result.message)
    if result.success:
        print(f"score={result.score}, position={result.position}")


if __name__ == "__main__":
    main()
