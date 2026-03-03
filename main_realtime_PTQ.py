import cv2
import numpy as np
import time
import openvino as ov
import csv, os, sys
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ====================== CONFIG ======================
MODEL_XML = "models/student_ptq.xml"
MODEL_BIN = "models/student_ptq.bin"
LABEL_PATH = "1_199_label.csv"

NUM_FRAMES = 8
IMAGE_SIZE = 224
CAPTURE_SECONDS = 5
CAMERA_FPS = 30
BUFFER_SIZE = CAPTURE_SECONDS * CAMERA_FPS   # ~150 frames

LOG_PATH = "ptq_button_log.txt"

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
        img = img.transpose(2,0,1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        clip.append(img)

    x = np.stack(clip, axis=1)
    return np.expand_dims(x.astype(np.float32), 0)


# ====================== LOAD MODEL (INT8 PTQ) ======================
core = ov.Core()
model = core.read_model(MODEL_XML, MODEL_BIN)
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

print("Model loaded (INT8 PTQ).")


# ====================== OPEN CAMERA ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera.")
    sys.exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera đã sẵn sàng. Nhấn phím 'c' để nhận diện 5 giây trước.")


# ====================== BUFFER ======================
buffer_frames = deque(maxlen=BUFFER_SIZE)


# ====================== FONT ======================
def get_font(size=32):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

font_main = get_font(32)
font_small = get_font(22)


# ====================== LOG ======================
log_f = open(LOG_PATH, "w")
log_f.write("timestamp,label,confidence\n")


# ====================== MAIN LOOP ======================
result_label = "..."
result_conf  = 0.0

last_time = time.time()

cv2.namedWindow("UsimKD - INT8 PTQ - Button Mode", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("UsimKD - INT8 PTQ - Button Mode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    buffer_frames.append(frame.copy())

    # FPS UI
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now

    key = cv2.waitKey(1) & 0xFF

    # ================== BẤM PHÍM “C” → NHẬN DIỆN ==================
    if key == ord('c'):
        print("Đang nhận diện 5 giây trước bằng mô hình INT8 PTQ...")

        if len(buffer_frames) < NUM_FRAMES:
            print("Không đủ dữ liệu để nhận diện.")
        else:
            total = len(buffer_frames)

            # Lấy đều NUM_FRAMES từ buffer 5 giây
            idxs = np.linspace(0, total - 1, NUM_FRAMES).astype(int)
            clip = [buffer_frames[i] for i in idxs]

            # RUN INFERENCE (PTQ)
            x = preprocess(clip)
            logits = compiled(x)[output_layer][0]

            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            idx = int(np.argmax(probs))
            result_label = LABELS[idx]
            result_conf = float(probs[idx])

            print(f"Kết quả: {result_label}  |  Confidence = {result_conf:.2f}")

            log_f.write(f"{time.time()},{result_label},{result_conf:.4f}\n")
            log_f.flush()

    # ================== DRAW UI ==================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img_pil)

    draw.text((20, 40), f"Kết quả: {result_label}", font=font_main, fill=(0,255,0))
    draw.text((20, 90), f"Độ tin cậy: {result_conf:.2f}", font=font_small, fill=(255,255,0))
    draw.text((20,140), f"FPS Camera: {fps:.1f}", font=font_small, fill=(255,255,255))
    draw.text((20,180), "Nhấn 'c' để nhận diện 5 giây trước", font=font_small, fill=(200,200,200))

    frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("UsimKD - INT8 PTQ - Button Mode", frame_show)

    if key == ord('q'):
        break

cap.release()
log_f.close()
cv2.destroyAllWindows()
