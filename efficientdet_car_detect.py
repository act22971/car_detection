import subprocess as sp
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from effdet import create_model
from effdet.efficientdet import HeadNet
import os
import time

print("🔥 SCRIPT START 🔥", flush=True)

# ตรวจสอบว่า GPU ใช้ได้
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ ใช้อุปกรณ์: {device}", flush=True)

# โหลด EfficientDet D0 pretrained บน COCO
model = create_model(
    model_name='efficientdet_d0',
    pretrained=True,
    num_classes=90,  # COCO = 91 classes (รวม background)
    bench_task='predict',
    image_size=(512, 512)
)

model.to(device).eval()

# กำหนดเฉพาะ ID ของรถใน COCO
TARGET_IDS = {
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    6: 'bus',
    8: 'truck'
}

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
    print("🚀 เริ่มสตรีม RTSP และประมวลผล EfficientDet ...", flush=True)
    pipe = ffmpeg_pipe(RTSP_URL)
    width, height = 640, 480

    # เตรียม Normalize transform
    normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    frame_count = 0
    try:
        while True:
            raw_frame = pipe.stdout.read(width * height * 3)
            if len(raw_frame) != width * height * 3:
                print("⚠️ ข้อมูลไม่ครบ — กล้องอาจหลุดการเชื่อมต่อ", flush=True)
                break

            # decode frame
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()

            # ดีบักว่าได้ frame จริง
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"   ประมวลผล frame ที่ {frame_count}", flush=True)
            # ปรับขนาดเป็น 512×512 แล้วแปลงเป็น Tensor
            img_resized = cv2.resize(frame, (512,512))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            tensor = T.ToTensor()(img_rgb).to(device)
            tensor = normalize(tensor)

            # inference
            tensor = tensor.unsqueeze(0)  # (1, 3, 512, 512)
            with torch.no_grad():
                outputs = model(tensor)

            # กรณีเป็น list ของ dicts
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    continue
                out = outputs[0]
                scores = out['scores'].detach().cpu().numpy()
                boxes = out['boxes'].detach().cpu().numpy()
                labels = out['labels'].detach().cpu().numpy()

            # กรณีเป็น tensor [N, 6]
            elif isinstance(outputs, torch.Tensor):
                preds = outputs[0].detach().cpu().numpy()
                boxes  = preds[:, :4]
                scores = preds[:, 4]
                labels = preds[:, 5].astype(int)

            else:
                print("❌ ไม่รู้จักโครงสร้างของผลลัพธ์:", type(outputs))
                continue

            keep = np.where(scores > 0.3)[0]
            for idx in keep:
                lbl = int(labels[idx])
                if lbl not in TARGET_IDS:
                    continue

                x1, y1, x2, y2 = boxes[idx].astype(int)
                cls_name = TARGET_IDS[lbl]
                conf = scores[idx]

                # scale กลับไปวาดบนภาพต้นฉบับขนาด 640×480
                # scale กลับไปวาดบนภาพต้นฉบับขนาด 640×480
                scale_x = width / 512    # 640 / 512 = 1.25
                scale_y = height / 512   # 480 / 512 ≈ 0.9375

                xx1, yy1 = int(x1 * scale_x), int(y1 * scale_y)
                xx2, yy2 = int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(frame, (xx1,yy1), (xx2,yy2), (0,255,0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}",
                            (xx1, yy1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # แสดงผล
            cv2.imshow("EffDet RTSP", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 ออกจากโปรแกรมแล้ว", flush=True)
                break

    finally:
        pipe.terminate()
        cv2.destroyAllWindows()
        print("✅ ปิดการเชื่อมต่อ FFmpeg และหน้าต่างแล้ว", flush=True)

if __name__ == "__main__":
    os.environ['FFREPORT'] = 'file=ffmpeg_errors.log'
    main()
