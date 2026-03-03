import cv2
import numpy as np
import time
import openvino as ov
import csv, os, sys
from collections import deque

# ====================== CONFIG ======================
MODEL_XML = "models/UsimKD_student_qat.xml"
LABEL_PATH = "1_199_label.csv"

NUM_FRAMES = 16              # CHUẨN TRAINING
IMAGE_SIZE = 224
CAPTURE_SECONDS = 3
CAMERA_FPS = 30
BUFFER_SIZE = CAPTURE_SECONDS * CAMERA_FPS

STABLE_WINDOW = 8            # stabilizer
STABLE_RATIO = 0.6

LOG_PATH = "qat_realtime_log.csv"

# ====================== LOAD LABELS ======================
LABELS = []
with open(LABEL_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        LABELS.append(row["name"].strip())

print("Loaded", len(LABELS), "labels.")

# ====================== PREPROCESS ======================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

def preprocess(frames):
    clip = []
    for f in frames:
        img = cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE))[:, :, ::-1] / 255.0
        img = (img.transpose(2,0,1) - IMAGENET_MEAN) / IMAGENET_STD
        clip.append(img)
    x = np.stack(clip, axis=1)   # (C,T,H,W)
    return np.expand_dims(x.astype(np.float32), 0)

# ====================== LOAD MODEL ======================
core = ov.Core()
model = core.read_model(MODEL_XML)
compiled = core.compile_model(
    model,
    "CPU",
    config={
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": "8",
        "NUM_STREAMS": "1"
    }
)
output_layer = compiled.output(0)

print("Model loaded (INT8 QAT).")

# ====================== CAMERA ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera.")
    sys.exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera đã sẵn sàng. Nhấn 'c' để nhận diện 3 giây trước.")

WINDOW_NAME = "UsimKD INT8 QAT Realtime"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ====================== BUFFER + STABILIZER ======================
buffer_frames = deque(maxlen=BUFFER_SIZE)
pred_window = deque(maxlen=STABLE_WINDOW)

# ====================== LOG FILE ======================
log_f = open(LOG_PATH, "w")
log_f.write("timestamp,label,confidence,fps\n")

# ====================== MAIN LOOP ======================
result_label = "..."
result_conf = 0.0
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    buffer_frames.append(frame.copy())

    # FPS
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now

    key = cv2.waitKey(1) & 0xFF

    # ================== BẤM PHÍM “C” → CHẤM 3 GIÂY TRƯỚC ==================
    if key == ord('c'):
        if len(buffer_frames) < NUM_FRAMES:
            print("Không đủ dữ liệu để nhận diện.")
        else:
            total = len(buffer_frames)

            # Motion-aware sampling: lấy 16 frame gần nhất
            start = total - NUM_FRAMES
            idxs = list(range(start, total))

            clip = [buffer_frames[i] for i in idxs]

            x = preprocess(clip)
            logits = compiled(x)[output_layer][0]

            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            idx = int(np.argmax(probs))
            label = LABELS[idx]
            conf  = float(probs[idx])

            pred_window.append((label, conf))

            # -------- Stabilizer (window voting) --------
            labels = [l for l, c in pred_window]
            confs  = [c for l, c in pred_window]

            candidate = max(set(labels), key=labels.count)
            freq = labels.count(candidate)
            avg_conf = sum(confs) / len(confs)

            if freq >= STABLE_WINDOW * STABLE_RATIO:
                result_label = candidate
                result_conf = avg_conf
            else:
                # still unstable
                result_label = label
                result_conf = conf

            print(f"Nhận diện: {result_label} | Conf={result_conf:.2f}")

            log_f.write(f"{time.time()},{result_label},{result_conf:.4f},{fps:.2f}\n")
            log_f.flush()

    # ================== DRAW UI ==================
    ui = frame.copy()
    cv2.putText(ui, f"Label: {result_label}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.putText(ui, f"Conf: {result_conf:.2f}", (40,140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
    cv2.putText(ui, f"FPS: {fps:.1f}", (40,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(ui, "Nhấn 'c' để nhận diện 3 giây trước", (40,260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

    cv2.imshow(WINDOW_NAME, ui)

    if key == ord('q'):
        break

log_f.close()
cap.release()
cv2.destroyAllWindows()
