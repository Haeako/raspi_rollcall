import sys
import os
import cv2
import time

# Thêm thư mục gốc vào sys.path để import các module dễ dàng hơn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.src.camera import PiCamera
from core.src.model import FaceModel, draw_polyboxes
from core.src.database import Qdrant_db

def main():
    print("[INFO] Khởi tạo Camera...")
    cam = PiCamera()
    
    print("[INFO] Khởi tạo Model...")
    # Cần chạy script này từ thư mục gốc của project (raspi_rollcall)
    # để các đường dẫn weight hoạt động đúng.
    face_model = FaceModel(
        det_weight="weights/det_10g.onnx",
        rec_weight="weights/w600k_r50.onnx"
    )
    
    print("[INFO] Khởi tạo Database...")
    # Chạy script ở ngoài docker nên host là localhost thay vì 'qdrant'
    db = Qdrant_db(host="qdrant", port=6333)
    
    print("[INFO] Bắt đầu chạy pipeline. Nhấn 'q' trên cửa sổ Camera để thoát.")
    
    try:
        while True:
            # 1. Bắt khung hình từ camera
            frame = cam.capture_array()
            
            # picamera2 thường trả về ảnh dạng RGB
            # 2. Nhận diện và lấy vector (FaceModel được thiết kế tự xử lý nếu truyền format='RGB')
            emb, bbs, ccs = face_model.get_embeding_vector(frame, image_format="RGB")
            
            # Chuyển frame sang BGR để hiển thị OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if len(emb) == 0:
                # cv2.imshow("Camera", bgr_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                    # break
                continue
            
            # 3. Lấy khuôn mặt đầu tiên phát hiện được
            target_emb = emb[0]
            target_bb = bbs[0]
            target_cc = ccs[0]
            
            # 4. Search vector trong CSDL
            results = db.search(target_emb, top_k=1, threshold=0.4)
            name, score = results[0]
            
            # Vẽ bounding box lên ảnh
            # bgr_frame = draw_polyboxes(bgr_frame, [target_bb], [target_cc], names=[name])
            # cv2.imshow("Camera", bgr_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                
            if name != "Unknown":
                print(f"[NHẬN DIỆN] Xin chào {name} (Độ tin cậy: {score:.2f})")
                time.sleep(1) # Tạm dừng 1 chút để tránh spam terminal liên tục
            else:
                # 5. Nếu không có trong cơ sở dữ liệu -> Tạm dừng và hỏi
                print(f"[CẢNH BÁO] Phát hiện người lạ (Score: {score:.2f})")
                print("==> Tạm dừng Camera để hỏi Enroll <==")
                
                # Hỏi ý kiến qua terminal
                ans = input("Bạn có muốn enroll khuôn mặt này không? (y/n): ")
                
                if ans.strip().lower() == 'y':
                    new_name = input("Nhập tên cho khuôn mặt này: ")
                    if new_name.strip():
                        # Lưu vào CSDL
                        db.add(target_emb, new_name.strip())
                        print(f"[THÀNH CÔNG] Đã lưu khuôn mặt của '{new_name.strip()}' vào CSDL.")
                    else:
                        print("[LỖI] Tên không hợp lệ, bỏ qua enroll.")
                else:
                    print("[INFO] Bỏ qua enroll.")
                
                print("[INFO] Trở lại bắt camera...\n")
                # Đợi một xíu để người dùng tránh camera nếu muốn, rồi chạy tiếp
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[INFO] Đã nhận tín hiệu dừng từ bàn phím...")
    finally:
        print("[INFO] Đang dọn dẹp...")
        cam.close()
        # cv2.destroyAllWindows()
        print("[INFO] Đã đóng Camera hoàn tất.")

if __name__ == "__main__":
    main()
