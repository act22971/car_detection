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
# โหลดโมเดล YOLOv8 (แนะนำเปลี่ยนเป็น yolov8n.pt หรือ yolov8s.pt หากช้าเกินไป)
model = YOLO('yolov8n.pt')
model.to(device)
# RTSP URL
RTSP_URL = 'rtsp://admin:NT2%40admin@ntcctvptn.totddns.com:64780/cam/realmonitor?channel=1&subtype=0'

def ffmpeg_pipe(url, width=640, height=480):
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

def main():
    print("🚀 เริ่มสตรีม RTSP และประมวลผล YOLOv8 ...")
    pipe = ffmpeg_pipe(RTSP_URL)
    width, height = 640, 480

    # สำหรับเก็บ track memory
    track_memory = {}

    try:
        while True:
            raw_frame = pipe.stdout.read(width * height * 3)
            if len(raw_frame) != width * height * 3:
                print("⚠️ ข้อมูลไม่ครบ — กล้องอาจหลุดการเชื่อมต่อ")
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

            # รัน YOLO tracking
            results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=1280,
            conf=0.15,  # ลดจาก 0.25
            iou=0.5,
            agnostic_nms=True
            )


            # กรองเฉพาะ class ที่ต้องการ
            target_classes = ['car', 'motorcycle', 'bus','truck', 'bicycle']
            names = model.names

            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    keep_idx = [i for i, cls_id in enumerate(cls_ids) if names[cls_id] in target_classes]
                    result.boxes = result.boxes[keep_idx] if keep_idx else None

            # บันทึกตำแหน่ง object ที่มี id
            current_time = time.time()
           # ลูปเช็ค boxes
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.id is not None:
                        id = int(box.id.item())
                        xyxy = box.xyxy.cpu().numpy()[0]
                        track_memory[id] = {
                        'last_seen': current_time,
                        'box': xyxy
                        }


            # ลบ object ที่หายไปนานเกิน 5 วิ
            track_memory = {
                k: v for k, v in track_memory.items()
                if current_time - v['last_seen'] < 5.0
            }

            # วาดกรอบผลลัพธ์
            output = results[0].plot()

            # แสดงผล
            cv2.imshow("YOLOv8 RTSP", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 ออกจากโปรแกรมแล้ว")
                break

    finally:
        pipe.terminate()
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อ FFmpeg และหน้าต่างแล้ว")

if __name__ == "__main__":
    os.environ['FFREPORT'] = 'file=ffmpeg_errors.log'
    main()
