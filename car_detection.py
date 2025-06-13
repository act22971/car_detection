import subprocess as sp
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import torch

# ตรวจสอบว่า GPU ใช้ได้หรือไม่
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ ใช้อุปกรณ์: {device}")

# โหลดโมเดล YOLOv8 ที่แม่นยำขึ้น
model = YOLO('yolov8s.pt').to(device)

# RTSP URL
RTSP_URL = 'rtsp://admin:NT2%40admin@ntcctvptn.totddns.com:64780/cam/realmonitor?channel=1&subtype=0'

# ความละเอียดของกล้อง
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# วัตถุที่ต้องการตรวจจับ
TARGET_CLASSES = {'car', 'motorcycle', 'person'}

def get_color_by_class(class_name):
    color_map = {#BGR
        'person': (0, 128, 255),       # แดง
        'car': (0, 255, 0),          # เขียว
        'motorcycle': (255, 0, 0), # ส้มอ่อน
    }
    return color_map.get(class_name, (255, 255, 255))  # สีขาวเป็นค่า default


def ffmpeg_pipe(url, width=1280, height=720):
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', url,
        '-loglevel', 'quiet',
        '-an',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-vcodec', 'rawvideo',
        '-'
    ]
    return sp.Popen(cmd, stdout=sp.PIPE, bufsize=10**8)

def preprocess_frame(frame):
    """เพิ่ม contrast และปรับแสง"""
    # frame = cv2.GaussianBlur(frame, (3, 3), 0)  # ใช้ถ้ากล้อง noisy
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

def main():
    print("🚀 เริ่มสตรีม RTSP และประมวลผล YOLOv8 ...")
    pipe = ffmpeg_pipe(RTSP_URL, FRAME_WIDTH, FRAME_HEIGHT)
    track_memory = {}
    cv2.namedWindow("YOLOv8 RTSP", cv2.WINDOW_NORMAL)

    try:
        while True:
            raw_frame = pipe.stdout.read(FRAME_WIDTH * FRAME_HEIGHT * 3)
            if len(raw_frame) != FRAME_WIDTH * FRAME_HEIGHT * 3:
                print("⚠️ ข้อมูลไม่ครบ — กล้องอาจหลุดการเชื่อมต่อ")
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
            frame = preprocess_frame(frame)

            results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=1280,
            conf=0.2,
            iou=0.45,
            agnostic_nms=True,
            device=device  # ให้มั่นใจว่าใช้ GPU ถ้ามี
            )


            current_time = time.time()
            names = model.names

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)

                    keep = []
                    for i, cls_id in enumerate(cls_ids):
                        class_name = names[cls_id] if cls_id < len(names) else ""
                        if class_name in TARGET_CLASSES:
                            keep.append(i)
                            # บันทึกตำแหน่ง object
                            if ids[i] is not None:
                                track_memory[ids[i]] = {
                                    'last_seen': current_time,
                                    'box': boxes.xyxy[i].cpu().numpy()
                                }

                    if keep:
                        result.boxes = result.boxes[keep]
                    else:
                        result.boxes = None

            # ลบ object ที่หายไปนานเกิน 5 วินาที
            track_memory = {
                k: v for k, v in track_memory.items()
                if current_time - v['last_seen'] < 15.0
            }

            # วาดผลลัพธ์
            output = frame.copy()
            if results[0].boxes is not None:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)
                xyxy = boxes.xyxy.cpu().numpy()

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = model.names[cls_ids[i]]
                    object_id = ids[i]

                    # วาดกรอบด้วยเส้นบาง
                    color = get_color_by_class(class_name)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)


                    label = f"{class_name}"
                    if object_id is not None:
                        label += f" ID#{object_id}"

                    # วาด label
                    cv2.putText(output, label, (x1, y1 - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



            # แสดงผล
            cv2.imshow("YOLOv8 RTSP", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 ออกจากโปรแกรมแล้ว")
                break

            pipe.stdout.flush()

    finally:
        pipe.terminate()
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อ FFmpeg และหน้าต่างแล้ว")

if __name__ == "__main__":
    os.environ['FFREPORT'] = 'file=ffmpeg_errors.log'
    main()
