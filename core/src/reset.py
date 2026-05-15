#!/usr/bin/env python3
import logging
from database import Qdrant_db
from AS608 import AS608_HAL

logging.basicConfig(level=logging.INFO)

def reset_all():
    print("Bắt đầu quá trình reset toàn bộ dữ liệu...")

    try:
        # Reset Database Qdrant (Faces & Users)
        print("\n--- Xóa Database Qdrant ---")
        db = Qdrant_db()
        db.clear()
        print("Đã xóa xong Database Qdrant.")
    except Exception as e:
        print(f"Lỗi khi xóa Database Qdrant: {e}")

    try:
        # Reset Cảm biến vân tay AS608
        print("\n--- Xóa Cảm biến vân tay AS608 ---")
        sensor = AS608_HAL()
        result = sensor.empty()
        print(result.message)
    except Exception as e:
        print(f"Lỗi khi khởi tạo/xóa cảm biến AS608: {e}")

    print("\nQuá trình reset hoàn tất.")

if __name__ == "__main__":
    reset_all()
