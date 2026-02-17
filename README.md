# Face Recognition System - Docker Setup
# Chạy máy host là Ubuntu, Windown cần phải setup port cho vite
## 🚀 Khởi động nhanh

```bash
docker-compose up -d
```

## 🌐 Truy cập ứng dụng

Sau khi chạy `docker-compose up`, kiểm tra logs để lấy URL:

```bash
docker-compose logs frontend
```

**Output mẫu:**
```
frontend  | 
frontend  |   VITE v7.3.1  ready in 166 ms
frontend  | 
frontend  |   ➜  Local:   http://localhost:5173/
frontend  |   ➜  Network: http://172.18.0.4:5173/  ← Dùng link này để truy cập
frontend  |   ➜  press h + enter to show help
```

### 📍 Các địa chỉ truy cập:

- **Frontend (Vite)**: 
  - Từ máy host: `http://localhost:5173`
  - Từ mạng local: `http://<NETWORK_IP>:5173` (xem trong logs)
  
- **Backend (Python)**: `http://localhost:8000`

- **Qdrant (Vector DB)**: 
  - REST API: `http://localhost:6333`
  - Dashboard: `http://localhost:6333/dashboard`
  - gRPC: `http://localhost:6334`

## 📁 Cấu trúc bên trong container

```
/workspace/
├── core/                    # Backend code
│   └── main.py
├── face-recognition/        # Frontend code (Vite + React)
├── data/
├── external/
└── requirements.txt
```

## 🛠️ Quản lý services

### Xem logs
```bash
# Tất cả services
docker-compose logs -f

# Chỉ frontend
docker-compose logs -f frontend

# Chỉ backend
docker-compose logs -f backend
```

### 3. Dừng services
```bash
docker-compose down
```

### 4. Dừng và xóa volumes
```bash
docker-compose down -v
```

### 5. Restart một service cụ thể
```bash
docker-compose restart backend
docker-compose restart frontend
```

### 6. Truy cập vào container
```bash
# Backend
docker-compose exec backend bash

# Frontend
docker-compose exec frontend bash

# Kiểm tra cấu trúc thư mục
docker-compose exec backend ls -la /workspace/
```

## Ports
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **Qdrant**: http://localhost:6333 (REST API), http://localhost:6334 (gRPC)

## Lưu ý
- Code đã có sẵn trong image `a8c9b4c9bace`
- Backend entry: `/workspace/core/main.py`
- Frontend folder: `/workspace/face-recognition/`
- Frontend sử dụng NVM với Node version 21
- Qdrant data được lưu trong Docker volume `qdrant_storage`
- Tất cả services được kết nối trong cùng network `app_network`

## Kiểm tra cấu trúc container
```bash
# Xem cấu trúc thư mục
docker run --rm -it a8c9b4c9bace ls -la /workspace/

# Kiểm tra backend folder
docker run --rm -it a8c9b4c9bace ls -la /workspace/core/

# Kiểm tra frontend folder  
docker run --rm -it a8c9b4c9bace ls -la /workspace/face-recognition/
```

## Troubleshooting

### Nếu backend không tìm thấy main.py
Kiểm tra đường dẫn chính xác:
```bash
docker-compose exec backend find /workspace -name "main.py"
```

### Nếu port bị conflict
Sửa ports trong `docker-compose.yml`:
```yaml
ports:
  - "EXTERNAL_PORT:INTERNAL_PORT"
```

### Nếu NVM không tìm thấy
```bash
docker-compose exec frontend bash -c "source ~/.nvm/nvm.sh && nvm list"
```

### Rebuild services
```bash
docker-compose up -d --force-recreate
```
