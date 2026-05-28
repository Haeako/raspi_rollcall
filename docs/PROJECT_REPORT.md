# Bao cao du an Raspi Rollcall

## 1. Thong tin chung

**Ten du an:** Raspi Rollcall  
**Muc tieu:** Xay dung he thong diem danh/xac thuc ra vao tren Raspberry Pi bang cach ket hop nhan dien khuon mat, van tay va dashboard quan tri.  
**Nen tang trien khai:** Raspberry Pi, Python, Flask, Qdrant, SQLite, camera Picamera2, cam bien HW-201 va cam bien van tay AS608/R307.  
**Thu muc du an:** `/home/raspi/Documents/raspi_rollcall`

Du an tap trung vao bai toan diem danh tai cho: khi co nguoi den gan cam bien, he thong tu dong chup anh, nhan dien khuon mat, quet van tay, danh gia ket qua va luu lich su. Cac truong hop chua ro rang hoac nguoi moi se duoc dua len dashboard de quan tri vien xu ly.

## 2. Pham vi chuc nang

He thong hien co cac nhom chuc nang chinh:

1. Tu dong kich hoat quy trinh diem danh khi cam bien HW-201 phat hien nguoi.
2. Chup anh bang camera Raspberry Pi va nhan dien khuon mat bang SCRFD + ArcFace.
3. Luu va tim kiem vector khuon mat trong Qdrant.
4. Quet, tim kiem, dang ky va xoa template van tay tren cam bien AS608/R307.
5. Ket hop diem van tay va do tin cay khuon mat bang Fuzzy Logic de dua ra quyet dinh.
6. Ghi nhan lich su diem danh, anh chup va cac trang thai can xu ly vao SQLite.
7. Cung cap dashboard web de xem thong ke, lich su, chi tiet anh chup, duyet nguoi moi va xu ly cac luot can xac minh.
8. Cung cap man hinh kiosk hien thi trang thai diem danh theo thoi gian gan thuc.
9. Ho tro reset sach du lieu, Qdrant va template van tay bang script rieng.

## 3. Cong nghe su dung

| Thanh phan | Cong nghe/thu vien | Vai tro |
| --- | --- | --- |
| Backend web | Flask | Dashboard, API, status screen |
| Luu lich su | SQLite | Luu attendance va pending faces |
| Vector database | Qdrant | Luu embedding khuon mat va vi tri van tay |
| AI khuon mat | SCRFD, ArcFace, ONNX Runtime | Phat hien mat va tao embedding |
| Camera | Picamera2 | Lay frame RGB tu camera Raspberry Pi |
| Van tay | pyfingerprint, AS608/R307 | Enroll, search, delete template |
| Cam bien kich hoat | gpiozero + lgpio, HW-201 | Phat hien nguoi de bat dau diem danh |
| Quyet dinh | scikit-fuzzy | Fuzzy Logic ket hop score/confidence |
| Giao dien | HTML, Bootstrap, Font Awesome, JavaScript | Dashboard quan tri va man hinh trang thai |
| Dong goi dich vu | Docker Compose | Chay Qdrant va backend container |

## 4. Cau truc thu muc

```text
raspi_rollcall/
├── server.py                         # Flask server, dashboard, API, khoi dong pipeline
├── attendance_store.py               # SQLite schema va ham ghi du lieu diem danh
├── app/
│   ├── app.py                        # Pipeline xu ly face + fingerprint + fuzzy
│   └── config.json                   # Cau hinh camera, sensor, threshold, Qdrant
├── core/
│   ├── paths.py                      # Quan ly duong dan du an, weights, data
│   └── src/
│       ├── AS608.py                  # Driver cam bien van tay AS608/R307
│       ├── HW_201.py                 # Driver cam bien vat can HW-201
│       ├── camera.py                 # Wrapper Picamera2
│       ├── database.py               # Client Qdrant bang HTTP API
│       └── model.py                  # FaceModel va FuzzyModel
├── templates/
│   ├── index.html                    # Dashboard quan tri
│   └── status_screen.html            # Man hinh kiosk trang thai
├── data/
│   ├── rollcall.db                   # SQLite database runtime
│   └── captures/                     # Anh chup diem danh
├── weights/
│   ├── det_10g.onnx                  # Model phat hien khuon mat
│   └── w600k_r50.onnx                # Model nhan dien khuon mat
├── docker_compose/
│   └── docker-compose.yml            # Qdrant va backend container
├── scripts/
│   ├── clean_reset.py                # Reset SQLite, captures, Qdrant, van tay
│   ├── install_kiosk_autostart.sh    # Cai autostart Chromium kiosk
│   └── rollcall-kiosk.desktop        # Desktop entry cho kiosk
├── external/face-reidentification/   # Submodule SCRFD/ArcFace
├── requirements.txt                  # Python dependencies
└── setup.bash                        # Cai goi he thong va tai weights
```

## 5. Kien truc tong the

He thong duoc chia thanh 4 lop:

1. **Lop phan cung:** HW-201, camera Raspberry Pi, AS608/R307.
2. **Lop xu ly sinh trac hoc:** FaceModel nhan dien khuon mat, AS608_HAL xu ly van tay, FuzzyModel dua ra quyet dinh.
3. **Lop luu tru:** Qdrant luu vector khuon mat; SQLite luu lich su diem danh, anh chup va hang cho duyet.
4. **Lop giao dien/API:** Flask server hien thi dashboard, status screen va cac API thao tac.

Luong du lieu tong quat:

```text
HW-201 detect
    -> Camera capture
    -> SCRFD detect face
    -> ArcFace create embedding
    -> Qdrant search
    -> AS608 search/enroll fingerprint
    -> Fuzzy decision
    -> SQLite attendance/pending_faces
    -> Dashboard/status screen
```

## 6. Mo ta luong diem danh

### 6.1. Trang thai pipeline

Trong `app/app.py`, pipeline chay theo state machine:

| Trang thai | Y nghia |
| --- | --- |
| `IDLE` | Cho cam bien HW-201 kich hoat |
| `WAIT_FACE` | Da kich hoat, dang cho worker khuon mat tra ket qua |
| `WAIT_FP` | Da co ket qua khuon mat, dang cho worker van tay |
| `VERIFY` | Co du lieu khuon mat va van tay, tien hanh doi chieu/quyet dinh |

### 6.2. Truong hop khuon mat da biet

1. HW-201 phat hien nguoi.
2. Face worker chup frame, xoay frame 180 do theo chieu doc, phat hien khuon mat va tao embedding.
3. Qdrant tim nguoi gan nhat theo cosine similarity.
4. Neu khuon mat khop nguoi da luu, he thong yeu cau AS608 quet van tay o che do `search`.
5. Pipeline lay `fingerprint_position` tra ve tu AS608 va tra nguoc ten tu Qdrant.
6. Neu ten theo khuon mat khac ten theo van tay, luot diem danh bi tu choi.
7. Neu khop, FuzzyModel tinh `decision` tu `fingerprint_score` va `face_confidence`.
8. He thong ghi vao SQLite:
   - `success` neu `decision >= 0.6`
   - `manual_review` neu `0.4 <= decision < 0.6`
   - `denied` neu `decision < 0.4`

### 6.3. Truong hop khuon mat moi

1. Face worker khong tim thay embedding du nguong trong Qdrant, dat ten la `Unknown`.
2. Pipeline chuyen sang enroll van tay thay vi search.
3. AS608 yeu cau quet van tay 2 lan, neu hop le se luu template va tra ve `fingerprint_position`.
4. He thong luu:
   - Mot ban ghi `pending_faces` gom embedding, anh chup, score, confidence va vi tri van tay.
   - Mot ban ghi `attendance` voi status `pending_unknown_face`.
5. Quan tri vien vao dashboard nhap ten nguoi dung va bam duyet.
6. Server them embedding vao Qdrant kem `fingerprint_position`, cap nhat pending face thanh `approved` va cap nhat attendance thanh `success`.
7. Neu bo qua, server xoa template van tay vua enroll va danh dau ban ghi lien quan la `denied`.

## 7. Cac module quan trong

### 7.1. `server.py`

`server.py` la entrypoint web chinh. File nay:

- Khoi tao Flask app.
- Khoi tao SQLite bang `init_attendance_db`.
- Khoi dong pipeline diem danh trong thread rieng bang `start_pipeline_once`.
- Doc thong ke, lich su va trang thai moi nhat de render dashboard.
- Cung cap API JSON cho dashboard va status screen.
- Xu ly duyet/tu choi manual review.
- Xu ly duyet/bo qua khuon mat moi.
- Phuc vu anh chup trong `data/captures`.

### 7.2. `app/app.py`

Day la pipeline xu ly chinh. File nay:

- Doc cau hinh tu `app/config.json`.
- Khoi tao HW-201, Qdrant va FuzzyModel.
- Chay `face_worker` va `fingerprint_worker` trong process rieng.
- Dieu phoi state machine `IDLE -> WAIT_FACE -> WAIT_FP -> VERIFY`.
- Luu anh chup diem danh co bounding box.
- Ghi ket qua vao SQLite thong qua `record_attendance` va `record_pending_face`.
- Xu ly shutdown va giai phong tai nguyen phan cung.

### 7.3. `attendance_store.py`

File nay quan ly SQLite local:

- Tao/migrate bang `attendance`.
- Tao/migrate bang `pending_faces`.
- Ghi mot luot diem danh bang `record_attendance`.
- Ghi khuon mat moi cho dashboard duyet bang `record_pending_face`.

### 7.4. `core/src/model.py`

File nay chua 2 model:

- `FaceModel`: dung SCRFD de detect mat, ArcFace de tao embedding 512 chieu.
- `FuzzyModel`: dung scikit-fuzzy de ket hop `fingerprint_score` va `face_confidence` thanh diem quyet dinh tu 0 den 1.

Neu `scikit-fuzzy` khong duoc cai, model co fallback tinh diem don gian theo ty trong:

```text
decision = fingerprint_score_norm * 0.55 + face_confidence * 0.45
```

### 7.5. `core/src/database.py`

`Qdrant_db` la wrapper HTTP API cho Qdrant:

- Tao collection `faces` neu chua co.
- Them embedding moi kem payload `name` va `fingerprint_position`.
- Tim kiem embedding theo cosine similarity.
- Lay ten theo vi tri van tay.
- Lay vi tri van tay theo ten.
- Xoa nguoi, clear collection, dem so point va liet ke du lieu.

### 7.6. `core/src/AS608.py`

Module nay dieu khien cam bien van tay AS608/R307 qua UART:

- Cong mac dinh: `/dev/ttyAMA0`
- Baudrate: `57600`
- `enroll`: dang ky van tay moi bang 2 lan quet.
- `search`: tim van tay trong bo nho cam bien.
- `delete`: xoa template theo vi tri.
- `empty`: xoa toan bo template.

### 7.7. `core/src/HW_201.py`

Driver HW-201 dung `gpiozero` voi `lgpio`, phu hop Raspberry Pi 5 va `/dev/gpiochip*`.

- Pin mac dinh: BCM 26.
- `active_state=False`, nghia la tin hieu LOW duoc xem la co vat can.
- `detect()` tra ve `True` khi co nguoi/vat can kich hoat.

### 7.8. `core/src/camera.py`

Wrapper Picamera2:

- Cau hinh RGB888, kich thuoc va framerate tu config.
- Khoi dong camera co retry 3 lan.
- Khi capture loi, thu restart camera roi capture lai.
- Tra ve frame RGB cho pipeline.

## 8. Thiet ke co so du lieu SQLite

SQLite database nam tai:

```text
data/rollcall.db
```

### 8.1. Bang `attendance`

| Cot | Kieu | Mo ta |
| --- | --- | --- |
| `id` | INTEGER PK | ID tu tang cua luot diem danh |
| `name` | TEXT | Ten nguoi duoc nhan dien, hoac `Unknown` |
| `time` | TEXT | Thoi gian ghi nhan |
| `status` | TEXT | `success`, `pending_unknown_face`, `manual_review`, `uncertain`, `denied` |
| `image_path` | TEXT | Ten file anh trong `data/captures` |
| `face_score` | REAL | Diem tu Qdrant search |
| `face_confidence` | REAL | Do tin cay detect khuon mat |
| `fingerprint_score` | REAL | Diem khop van tay tu AS608 |
| `decision` | REAL | Diem quyet dinh fuzzy |
| `note` | TEXT | Ghi chu xu ly |

### 8.2. Bang `pending_faces`

| Cot | Kieu | Mo ta |
| --- | --- | --- |
| `id` | INTEGER PK | ID tu tang |
| `time` | TEXT | Thoi gian phat hien |
| `image_path` | TEXT | Anh khuon mat can duyet |
| `embedding` | TEXT | Embedding JSON da serialize |
| `face_score` | REAL | Diem search gan nhat |
| `face_confidence` | REAL | Confidence cua face detector |
| `fingerprint_position` | INTEGER | Vi tri template van tay vua enroll |
| `status` | TEXT | `pending`, `approved`, `skipped` |
| `approved_name` | TEXT | Ten duoc quan tri vien gan khi duyet |
| `note` | TEXT | Ghi chu |

## 9. Thiet ke Qdrant

Collection mac dinh:

```text
faces
```

Cau hinh vector:

| Thuoc tinh | Gia tri |
| --- | --- |
| Vector size | 512 |
| Distance | Cosine |

Moi point trong Qdrant gom:

```json
{
  "id": 0,
  "vector": [/* ArcFace embedding */],
  "payload": {
    "name": "Nguyen Van A",
    "fingerprint_position": 3
  }
}
```

Lien ket quan trong la `fingerprint_position`: Qdrant khong chi luu khuon mat, ma con anh xa nguoi dung voi template van tay trong bo nho AS608.

## 10. API va giao dien web

### 10.1. Route HTML

| Method | Route | Mo ta |
| --- | --- | --- |
| GET | `/` | Dashboard quan tri |
| GET | `/status-screen` | Man hinh kiosk trang thai |
| GET | `/captures/<filename>` | Lay anh chup diem danh |

### 10.2. API JSON

| Method | Route | Mo ta |
| --- | --- | --- |
| GET | `/api/stats` | Lay thong ke tong, thanh cong, can xu ly, tu choi |
| GET | `/api/status` | Lay trang thai diem danh moi nhat |
| GET | `/api/attendance/<id>` | Lay chi tiet mot luot diem danh |
| POST | `/api/attendance/<id>/approve` | Duyet luot manual review |
| POST | `/api/attendance/<id>/deny` | Tu choi luot manual review |
| POST | `/api/pending_faces/<id>/approve` | Them nguoi moi vao Qdrant va cap nhat attendance |
| POST | `/api/pending_faces/<id>/skip` | Bo qua khuon mat moi va xoa template van tay |

### 10.3. Dashboard `templates/index.html`

Dashboard hien thi:

- Thanh thong ke tong luot, thanh cong, can xu ly, tu choi va so dong lich su dang hien thi.
- Panel trang thai diem danh moi nhat.
- Danh sach khuon mat moi can duyet, kem anh, score, confidence va vi tri van tay.
- Bang lich su diem danh.
- Modal chi tiet gom anh chup, ten, thoi gian, status, face score, face confidence, fingerprint score, fuzzy decision va ghi chu.

### 10.4. Status screen `templates/status_screen.html`

Man hinh kiosk toi gian cho nguoi dung dung truoc thiet bi:

- Hien thi dong ho.
- Hien thi trang thai moi nhat tu `/api/status`.
- Doi mau theo state: idle, ok, review, denied, error.
- Tu dong refresh moi 1.5 giay.

## 11. Cau hinh he thong

File cau hinh chinh:

```text
app/config.json
```

Cac tham so dang duoc dat:

| Tham so | Gia tri hien tai | Mo ta |
| --- | --- | --- |
| `trigger` | `sensor` | Kieu kich hoat pipeline |
| `sensor_pin` | `26` | BCM pin cua HW-201 |
| `sensor_cooldown` | `3.0` | Thoi gian chong kich hoat lap lai |
| `camera_width` | `320` | Chieu rong frame |
| `camera_height` | `240` | Chieu cao frame |
| `camera_framerate` | `10` | FPS camera |
| `camera_buffer_count` | `2` | So buffer camera |
| `face_confidence_thresh` | `0.5` | Nguong confidence detect face |
| `face_threshold` | `0.4` | Nguong score search Qdrant |
| `face_timeout` | `5.0` | Timeout cho face worker |
| `fp_timeout` | `10.0` | Timeout cho fingerprint worker |
| `worker_startup_timeout` | `90.0` | Timeout cho worker san sang |
| `det_weight` | `det_10g.onnx` | Model detect face |
| `rec_weight` | `w600k_r50.onnx` | Model recognition |
| `qdrant_host` | `localhost` | Host Qdrant khi chay local |
| `qdrant_port` | `6333` | Port HTTP Qdrant |

Bien moi truong `QDRANT_HOST` va `QDRANT_PORT` co the override cau hinh trong file.

## 12. Trien khai va chay he thong

### 12.1. Cai dat goi he thong

Script `setup.bash` cai cac goi can cho camera, GPIO, Flask, Picamera2, OpenCV va cac thu vien Raspberry Pi:

```bash
sudo bash setup.bash
```

Luu y: trong script hien tai co dong `cd weight`, trong khi thu muc cua du an la `weights`. Khi cai dat moi, can kiem tra lai va sua thanh `cd weights` neu can tai weight tu script.

### 12.2. Cai Python dependencies

```bash
pip install -r requirements.txt
```

### 12.3. Chay Qdrant bang Docker Compose

```bash
docker compose -f docker_compose/docker-compose.yml up -d qdrant
```

Neu chay backend trong container:

```bash
docker compose -f docker_compose/docker-compose.yml up -d
docker exec -it backend bash
```

### 12.4. Chay Flask server va pipeline

Tai thu muc goc du an:

```bash
python server.py
```

Server mac dinh lang nghe tai:

```text
http://0.0.0.0:5000
```

Dashboard:

```text
http://<raspberry-pi-ip>:5000/
```

Status screen:

```text
http://<raspberry-pi-ip>:5000/status-screen
```

## 13. Kiosk autostart

Du an co script cai Chromium kiosk:

```bash
sh scripts/install_kiosk_autostart.sh
```

Script nay tao desktop entry tai:

```text
/home/raspi/.config/autostart/rollcall-kiosk.desktop
```

Khi desktop session khoi dong, Chromium se mo:

```text
http://127.0.0.1:5000/status-screen
```

## 14. Reset du lieu

Script reset:

```bash
python scripts/clean_reset.py
```

Mac dinh script se yeu cau nhap `RESET` de xac nhan, vi thao tac nay xoa:

- SQLite database `data/rollcall.db`
- Anh trong `data/captures`
- Qdrant collection `faces`
- Toan bo template van tay tren AS608

Mot so tuy chon:

```bash
python scripts/clean_reset.py --yes
python scripts/clean_reset.py --skip-fingerprint
python scripts/clean_reset.py --skip-qdrant
python scripts/clean_reset.py --qdrant-host localhost --qdrant-port 6333
```

## 15. Yeu cau phan cung

| Thiet bi | Vai tro | Ghi chu |
| --- | --- | --- |
| Raspberry Pi | May chu xu ly | Can ho tro camera, GPIO, UART |
| Camera Pi | Chup anh khuon mat | Dung Picamera2 |
| HW-201 | Cam bien kich hoat | Pin mac dinh BCM 26 |
| AS608/R307 | Van tay | UART `/dev/ttyAMA0`, baud 57600 |
| Man hinh | Dashboard/kiosk | Co the dung Chromium kiosk |

Neu chay trong Docker, compose da mount cac thiet bi quan trong nhu `/dev/video*`, `/dev/gpiochip0`, `/dev/gpiomem`, `/dev/i2c-1`, `/dev/spidev*` va `/dev`.

## 16. Xu ly loi va han che hien tai

Mot so diem can luu y:

1. `setup.bash` dang `cd weight` nhung thu muc trong repo la `weights`; can chinh khi dung script cai dat moi.
2. Pipeline phu thuoc phan cung that, nen can kiem tra quyen truy cap camera, UART va GPIO khi chay bang Docker.
3. `Qdrant_db._next_id()` dung `count()` lam ID moi. Neu co xoa point rieng le, ID co the bi trung trong mot so truong hop; nen can nhac dung UUID hoac bo dem rieng neu he thong mo rong.
4. `server.py` khoi dong pipeline trong thread khi chay Flask. Neu dung WSGI multi-worker, co nguy co nhieu pipeline cung chay; cach chay hien tai phu hop Flask single process.
5. Dashboard dang dung CDN Bootstrap/Font Awesome, can internet de tai CSS/JS neu chay lan dau hoac khong co cache.
6. Chua thay test tu dong trong repo; viec kiem thu hien tai chu yeu la kiem thu thu cong voi phan cung.

## 17. De xuat kiem thu

### 17.1. Kiem thu module

- Kiem tra `init_attendance_db` tao dung bang va cot.
- Kiem tra `record_attendance` ghi du cac truong.
- Kiem tra `record_pending_face` serialize embedding dung JSON.
- Mock Qdrant API de kiem tra `add`, `search`, `get_name_by_fp_position`.
- Kiem tra FuzzyModel voi cac cap score/confidence dai dien.

### 17.2. Kiem thu tich hop

- Chay Qdrant, them mot embedding mau va search lai.
- Test camera capture anh vao thu muc tam.
- Test AS608 enroll, search, delete voi mot ngon tay mau.
- Test HW-201 detect khi co/khong co vat can.
- Test dashboard duyet pending face va xac nhan Qdrant co point moi.

### 17.3. Kiem thu nghiem thu

Kich ban can kiem:

1. Nguoi da dang ky, mat va van tay khop: ket qua `success`.
2. Nguoi da dang ky, mat dung nhung van tay nguoi khac: ket qua `denied`.
3. Khuon mat moi: tao `pending_unknown_face`, dashboard duyet thanh cong.
4. Khuon mat moi bi bo qua: xoa template van tay va danh dau `denied`.
5. Khong thay mat trong timeout: pipeline quay lai `IDLE`.
6. Khong quet van tay trong timeout: pipeline quay lai `IDLE`.

## 18. Huong phat trien

1. Them authentication cho dashboard quan tri.
2. Them trang quan ly nguoi dung: danh sach, sua ten, xoa nguoi, cap nhat van tay.
3. Them API export lich su diem danh ra CSV/Excel.
4. Them test tu dong cho SQLite, Qdrant wrapper va Flask API.
5. Tach pipeline thanh service rieng thay vi thread trong Flask server.
6. Them logging co cau truc va luu log ra file.
7. Dong bo trang thai realtime bang WebSocket/SSE thay vi polling.
8. Cai tien ID cua Qdrant point de tranh trung khi xoa/them du lieu.
9. Luu nhieu embedding cho mot nguoi de tang do on dinh nhan dien.
10. Them co che backup/restore SQLite, Qdrant va template van tay.

## 19. Ket luan

Raspi Rollcall la mot he thong diem danh sinh trac hoc kha day du cho moi truong Raspberry Pi. Du an da ket hop duoc phan cung that, nhan dien AI, vector database, fuzzy decision va dashboard quan tri trong mot luong xu ly khep kin. Diem manh cua thiet ke la co co che xu ly nguoi moi thong qua `pending_faces`, luu anh minh chung cho tung luot diem danh va doi chieu cheo giua khuon mat voi van tay.

De dua vao van hanh on dinh hon, nen uu tien bo sung bao mat dashboard, test tu dong, quan ly nguoi dung day du va tach pipeline thanh service doc lap. Nhung voi cau truc hien tai, du an da co nen tang tot de trien khai mot kiosk diem danh/xac thuc tren Raspberry Pi.
