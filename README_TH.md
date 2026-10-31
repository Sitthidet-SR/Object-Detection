# Stationery Detector (YOLO + OpenCV + PyQt6)

โปรแกรม Desktop ตรวจจับและจำแนกเครื่องเขียน 5 ชนิดผ่าน Webcam แบบ Real-time
ได้แก่ ปากกา (Pen), ดินสอ (Pencil), ยางลบ (Eraser), ไม้บรรทัด (Ruler), กรรไกร (Scissors)

## ฟังก์ชันของโปรแกรม

| ฟังก์ชัน | รายละเอียด |
|---|---|
| Live Camera | แสดงภาพจาก Webcam (เลือกหมายเลขกล้องได้) |
| Object Detection | ตรวจจับวัตถุด้วย YOLO (เปิด/ปิดการตรวจจับได้) |
| Object Classification | จำแนกเป็น 5 ชนิด แต่ละชนิดมีสีกรอบต่างกัน |
| Bounding Box | วาดกรอบรอบวัตถุพร้อมชื่อคลาส |
| Confidence Score | แสดงค่าความมั่นใจบนกรอบและในรายการด้านขวา ปรับเกณฑ์ขั้นต่ำด้วยแถบเลื่อน |
| Object Counter | นับจำนวนวัตถุแต่ละชนิดและผลรวม พร้อม FPS |
| Capture Image | กดปุ่ม "บันทึกภาพ" หรือ Spacebar เพื่อบันทึกภาพผลลัพธ์ลง `captures/` (ได้ไฟล์ .jpg และ .json ที่เก็บผลการตรวจจับ) |

## โครงสร้างไฟล์

```
app.py                 หน้าต่างโปรแกรม (PyQt6) และ thread อ่านกล้อง
detector.py            ตัวโหลดโมเดล YOLO, วาดกรอบ, นับจำนวน
train.py               สคริปต์เทรนโมเดล
data.yaml              ตั้งค่า dataset 5 คลาส
tools/collect_images.py  ถ่ายภาพจากกล้องเพื่อเก็บ dataset
tools/split_dataset.py   แบ่งภาพที่ติด label แล้วเป็น train/val
models/best.pt         โมเดลที่เทรนแล้ว (สร้างหลังเทรน)
captures/              ภาพที่บันทึกจากโปรแกรม
```

## 1. ติดตั้ง

```bash
cd "/home/jackit/MCRU/Object Detection"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ถ้ามี GPU NVIDIA ให้ติดตั้ง PyTorch รุ่น CUDA ตามคำแนะนำที่ pytorch.org ก่อน จะเทรนและรันได้เร็วขึ้นมาก

## 2. รันโปรแกรม

```bash
python app.py
```

**สำคัญ:** โมเดล YOLO สำเร็จรูป (COCO) รู้จักแค่ "กรรไกร" ไม่รู้จักปากกา ดินสอ ยางลบ ไม้บรรทัด
ถ้ายังไม่มี `models/best.pt` โปรแกรมจะใช้โมเดล COCO แทนและตรวจได้เฉพาะกรรไกร
ต้องเทรนโมเดลเองตามขั้นตอนด้านล่างจึงจะตรวจได้ครบ 5 ชนิด
(หรือกดปุ่ม "เลือกไฟล์โมเดล" เพื่อเลือกไฟล์ .pt ที่อื่นก็ได้)

## 3. เตรียม Dataset

### 3.1 เก็บภาพ
```bash
python tools/collect_images.py --out dataset/raw
```
กด 1-5 เพื่อบันทึกภาพของแต่ละชนิด, 0 สำหรับภาพที่มีหลายชิ้นปนกัน, q เพื่อออก

คำแนะนำ:
- อย่างน้อย **150-300 ภาพต่อคลาส** (ยิ่งมากยิ่งแม่น)
- ถ่ายหลายมุม หลายระยะ หลายพื้นหลัง หลายสภาพแสง และใช้ของหลายแบบ/หลายสี
- ควรมีภาพที่มีวัตถุหลายชนิดในภาพเดียว เพื่อให้แยกปากกากับดินสอได้ดีขึ้น
- สามารถใช้ dataset สาธารณะจาก Roboflow Universe เสริมได้ (ค้นคำว่า pen, pencil, eraser, ruler, scissors แล้ว export เป็น "YOLOv8") แต่ต้องตรวจให้ลำดับคลาสตรงกับ `data.yaml`

### 3.2 ติด Label (YOLO format)
ใช้เครื่องมือใดก็ได้: [Roboflow](https://roboflow.com), [CVAT](https://cvat.ai), [Label Studio](https://labelstud.io) หรือ LabelImg
ตั้งชื่อคลาสและลำดับให้ตรงกันดังนี้:

| id | คลาส |
|---|---|
| 0 | Pen |
| 1 | Pencil |
| 2 | Eraser |
| 3 | Ruler |
| 4 | Scissors |

ไฟล์ label แต่ละไฟล์ (`.txt` ชื่อเดียวกับภาพ) มี 1 บรรทัดต่อ 1 วัตถุ:
```
<class_id> <x_center> <y_center> <width> <height>     # ค่าทั้งหมด normalize เป็น 0-1
```

### 3.3 แบ่ง train / val
ถ้า export จาก Roboflow มาเป็นโฟลเดอร์ train/valid อยู่แล้ว ให้จัดให้อยู่ตามโครงสร้างด้านล่าง
ถ้ามีภาพ+label รวมกันในโฟลเดอร์เดียว ใช้:
```bash
python tools/split_dataset.py --src labeled --dst dataset --val 0.2
```
ผลลัพธ์:
```
dataset/
  images/train/  images/val/
  labels/train/  labels/val/
```

## 4. เทรนโมเดล

```bash
python train.py                                   # yolov8n, 100 epochs
python train.py --model yolov8s.pt --epochs 150   # แม่นขึ้นแต่ช้าลง
python train.py --device cpu --batch 8            # ไม่มี GPU
```
เมื่อเทรนเสร็จ ไฟล์ `runs/stationery/weights/best.pt` จะถูกคัดลอกไปที่ `models/best.pt` อัตโนมัติ
ดูกราฟผลการเทรน (loss, mAP, confusion matrix) ได้ใน `runs/stationery/`
จากนั้นรัน `python app.py` ได้เลย

## 5. แก้ปัญหาที่พบบ่อย

- **เปิดกล้องไม่ได้**: ลองเปลี่ยน "กล้องหมายเลข" เป็น 1 หรือ 2 / บน Linux ตรวจว่าผู้ใช้อยู่ในกลุ่ม `video` (`sudo usermod -aG video $USER`)
- **Qt error เรื่อง xcb plugin**: `sudo apt install libxcb-cursor0`
- **ถ้าใช้ opencv-python แล้ว Qt ชนกัน**: ลอง `pip uninstall opencv-python && pip install opencv-python-headless` (แต่ `tools/collect_images.py` ต้องใช้ตัวเต็มเพราะเปิดหน้าต่าง OpenCV)
- **ปากกากับดินสอสับสนกัน**: เพิ่มภาพที่เห็นปลายปากกา/ไส้ดินสอชัด ๆ และเทรนด้วยโมเดลใหญ่ขึ้น (`yolov8s.pt`)
- **ช้า (FPS ต่ำ)**: ใช้ `yolov8n` และ `--imgsz 480` หรือใช้ GPU
