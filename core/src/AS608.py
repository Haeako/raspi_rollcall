#!/usr/bin/env python
"""
AS608_HAL – driver cho cảm biến vân tay AS608 / R307 qua UART.

Thay đổi so với bản gốc:
  - ReturnValue thêm field `score: int` để FuzzyWorker đọc trực tiếp
    mà không cần parse string.
  - __init__ không dùng `return` (constructor không return được giá trị);
    thay bằng raise RuntimeError khi init thất bại để caller biết.
  - search() trả về score thực từ sensor, không chỉ nhúng trong message.
  - Toàn bộ public method đều trả về ReturnValue nhất quán.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from pyfingerprint.pyfingerprint import PyFingerprint


# ---------------------------------------------------------------------------
# Data class trả về từ mọi thao tác
# ---------------------------------------------------------------------------

@dataclass
class FingerprintResult:
    success:  bool
    message:  str
    score:    int = 0          # match score (0 nếu không có)
    position: int = -1         # vị trí template trong sensor (-1 nếu không có)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class AS608_HAL:
    """
    Hardware Abstraction Layer cho cảm biến vân tay AS608.

    Raises:
        RuntimeError: nếu không kết nối được hoặc sai password.
    """

    UART_PORT     = "/dev/ttyAMA0"
    BAUD_RATE     = 57600
    ADDRESS       = 0xFFFFFFFF
    PASSWORD      = 0x00000000

    def __init__(
        self,
        port:     str = UART_PORT,
        baud:     int = BAUD_RATE,
        address:  int = ADDRESS,
        password: int = PASSWORD,
    ):
        try:
            self.sensor = PyFingerprint(port, baud, address, password)
        except Exception as e:
            raise RuntimeError(
                f"Không thể mở cổng UART '{port}': {e}"
            ) from e

        if not self.sensor.verifyPassword():
            raise RuntimeError(
                "Sai password cảm biến vân tay. Kiểm tra lại ADDRESS/PASSWORD."
            )

        count    = self.sensor.getTemplateCount()
        capacity = self.sensor.getStorageCapacity()
        print(f"[AS608] Kết nối OK – Templates: {count}/{capacity}")

    # -----------------------------------------------------------------------
    # Enroll – đăng ký vân tay mới (2 lần quét)
    # -----------------------------------------------------------------------

    def enroll(self, timeout: float = 0.0) -> FingerprintResult:
        """
        Đăng ký vân tay mới vào sensor.

        Returns:
            FingerprintResult với position = vị trí lưu nếu thành công.
        """
        try:
            print("[AS608] Đặt ngón tay lần 1...")
            start_time = time.time()
            while not self.sensor.readImage():
                if timeout > 0 and time.time() - start_time > timeout:
                    return FingerprintResult(success=False, message="Timeout khi chờ vân tay lần 1.")
                time.sleep(0.05)

            self.sensor.convertImage(0x01)

            # Kiểm tra đã enroll chưa
            pos, _ = self.sensor.searchTemplate()
            if pos >= 0:
                return FingerprintResult(
                    success=False,
                    message=f"Vân tay đã tồn tại tại vị trí #{pos}.",
                    position=pos,
                )

            print("[AS608] Nhấc ngón tay... Đặt lại sau 2 giây.")
            time.sleep(2)

            print("[AS608] Đặt ngón tay lần 2...")
            start_time = time.time()
            while not self.sensor.readImage():
                if timeout > 0 and time.time() - start_time > timeout:
                    return FingerprintResult(success=False, message="Timeout khi chờ vân tay lần 2.")
                time.sleep(0.05)

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

        except Exception as e:
            return FingerprintResult(
                success=False,
                message=f"Enroll thất bại: {e}",
            )

    # -----------------------------------------------------------------------
    # Search – nhận diện vân tay
    # -----------------------------------------------------------------------

    def search(self, timeout: float = 0.0) -> FingerprintResult:
        """
        Đọc vân tay và tìm trong database.

        Args:
            timeout: Thời gian chờ tối đa. 0.0 là chờ vô hạn.

        Returns:
            FingerprintResult với:
              - score    = accuracy score từ sensor (0–300+)
              - position = vị trí template khớp (-1 nếu không khớp)
        """
        try:
            start_time = time.time()
            while not self.sensor.readImage():
                if timeout > 0 and time.time() - start_time > timeout:
                    return FingerprintResult(
                        success=False,
                        message="Timeout: Không thấy vân tay.",
                        score=0,
                        position=-1,
                    )
                time.sleep(0.05)

            self.sensor.convertImage(0x01)

            position, score = self.sensor.searchTemplate()

            if position == -1:
                return FingerprintResult(
                    success=False,
                    message="Không tìm thấy vân tay khớp.",
                    score=0,
                    position=-1,
                )

            return FingerprintResult(
                success=True,
                message=(
                    f"Khớp tại vị trí #{position} "
                    f"với độ chính xác {score}."
                ),
                score=int(score),
                position=int(position),
            )

        except Exception as e:
            return FingerprintResult(
                success=False,
                message=f"Search thất bại: {e}",
                score=0,
            )

    # -----------------------------------------------------------------------
    # Delete – xoá một template
    # -----------------------------------------------------------------------

    def delete(self, position: int) -> FingerprintResult:
        """Xoá template tại `position`."""
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
        except Exception as e:
            return FingerprintResult(
                success=False,
                message=f"Delete thất bại: {e}",
            )

    # -----------------------------------------------------------------------
    # Empty – xoá toàn bộ database
    # -----------------------------------------------------------------------

    def empty(self) -> FingerprintResult:
        """Xoá toàn bộ database trong sensor."""
        try:
            if self.sensor.clearDatabase():
                return FingerprintResult(
                    success=True,
                    message="Đã xoá toàn bộ database.",
                )
            return FingerprintResult(
                success=False,
                message="Không thể xoá database.",
            )
        except Exception as e:
            return FingerprintResult(
                success=False,
                message=f"Empty thất bại: {e}",
            )

    # -----------------------------------------------------------------------
    # Template count
    # -----------------------------------------------------------------------

    def count(self) -> int:
        """Trả về số template đang lưu."""
        return self.sensor.getTemplateCount()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def main():
    try:
        hal = AS608_HAL()
    except RuntimeError as e:
        print(f"[ERROR] Không khởi tạo được sensor: {e}")
        return

    print("=== Enroll ===")
    result = hal.enroll()
    print(result.message)

    time.sleep(2)

    print("=== Search ===")
    result = hal.search()
    print(result.message)
    if result.success:
        print(f"  → score={result.score}, position={result.position}")

    print("=== Delete vị trí 0 ===")
    result = hal.delete(0)
    print(result.message)


if __name__ == "__main__":
    main()