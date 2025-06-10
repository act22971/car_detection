
import subprocess as sp
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os
import time

# โหลดโมเดล YOLOv8
model = YOLO('yolov8m.pt')

# RTSP URL ที่ใช้เข้ากล้อง (โค้ดเข้ารหัส @ ในรหัสผ่าน)
RTSP_URL = 'rtsp://admin:NT2%40admin@ntcctvptn.totddns.com:64780/cam/realmonitor?channel=1&subtype=0'

def ffmpeg_pipe(url, width=640, height=480):
    """
    ใช้ ffmpeg ส่ง raw BGR24 frames ผ่าน pipe ไปยัง stdout
    """
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',  # ใช้ TCP ลด packet loss
        '-i', url,
        '-loglevel', 'quiet',
        '-an',  # ไม่ดึง audio
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-vcodec', 'rawvideo',
        '-'
    ]
    return sp.Popen(cmd, stdout=sp.PIPE, bufsize=10**8)

def main():
    st.title("🚗 RTSP + FFmpeg + YOLOv8 ใน Streamlit")
    if st.button("▶️ เริ่มสตรีม"):
        pipe = ffmpeg_pipe(RTSP_URL)
        stframe = st.empty()

        try:
            while True:
                # อ่าน raw frame จาก pipe.stdout
                raw_frame = pipe.stdout.read(640 * 480 * 3)
                if len(raw_frame) != 640 * 480 * 3:
                    st.warning("✖️ ข้อมูล frame ไม่ครบ — stream อาจปิด")
                    break

                frame = np.frombuffer(raw_frame, np.uint8).reshape((480, 640, 3))

                # รัน infer YOLO
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=800,agnostic_nms=True)

                # Filter classes
                target_classes = ['car', 'motorcycle', 'bus']
                names = model.names  # e.g., {2: 'car', 3: 'motorcycle', 5: 'bus'}

                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                        keep_idx = [i for i, cls_id in enumerate(cls_ids) if names[cls_id] in target_classes]
                        if keep_idx:
                            result.boxes = result.boxes[keep_idx]
                        else:
                            result.boxes = None  # No boxes to keep
                # สร้าง dict เก็บตำแหน่ง object ตาม id
                if 'track_memory' not in st.session_state:
                    st.session_state.track_memory = {}

                # เก็บ ID เดิมไว้
                for box in results[0].boxes:
                    id = int(box.id.item())
                    xyxy = box.xyxy.cpu().numpy()[0]

                    st.session_state.track_memory[id] = {
                        'last_seen': time.time(),
                        'box': xyxy
                    }

                # ลบ id ที่หายไปนาน
                current_time = time.time()
                st.session_state.track_memory = {
                    k: v for k, v in st.session_state.track_memory.items()
                    if current_time - v['last_seen'] < 5.0  # 5 วิ
                }

                output = results[0].plot()


                # แสดงผลผ่าน Streamlit
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                stframe.image(output, use_container_width=True)

                time.sleep(0.03)

        finally:
            pipe.terminate()
            st.success("✅ ปิด FFmpeg และ stream แล้ว")

if __name__ == "__main__":
    os.environ['FFREPORT'] = 'file=ffmpeg_errors.log'  # บันทึก log สำหรับ debug
    main()
