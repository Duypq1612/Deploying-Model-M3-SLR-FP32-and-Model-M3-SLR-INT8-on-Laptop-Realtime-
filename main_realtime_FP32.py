import cv2, numpy as np, openvino as ov, time, csv
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ==== CONFIG ====
MODEL_PATH = "models/UsimKD_student_fp32.onnx"
LABEL_PATH = "1_199_label.csv"

IMAGE_SIZE = 224
BUFFER_SECONDS = 3
CAMERA_FPS = 60
BUFFER_SIZE = BUFFER_SECONDS * CAMERA_FPS   # ~150 frames
NUM_FRAMES = 16  # UsimKD input
LOG_PATH = "usimkd_capture_log.txt"

# ==== PREPROCESS ====
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

def preprocess(frames):
    clip = []
    for f in frames:
        img = cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE))[:, :, ::-1] / 255.0
        img = img.transpose(2,0,1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        clip.append(img)

    x = np.stack(clip, axis=1)  # (C, T, H, W)
    return np.expand_dims(x.astype(np.float32), 0)

# ==== LOAD LABEL ====
LABELS = []
with open(LABEL_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        LABELS.append(row["name"].strip())
print("Loaded", len(LABELS), "labels")

# ==== LOAD MODEL ====
core = ov.Core()
model = core.read_model(MODEL_PATH)
compiled = core.compile_model(model, "CPU", config={"PERFORMANCE_HINT": "LATENCY"})
output_layer = compiled.output(0)

print("Model loaded (OpenVINO FP32)")

# ==== LOG ====
log_file = open(LOG_PATH, "w")
log_file.write("timestamp,label,confidence\n")

# ==== BUFFER LƯU 5 GIÂY VIDEO ====
buffer_frames = deque(maxlen=BUFFER_SIZE)

# ==== CAMERA ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

print("🎥 Camera ready.")
print("Nhấn phím 'c' để CHẤM ĐIỂM động tác 5 giây trước.")
print("Nhấn 'q' để thoát.")

font_main = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)

result_label = "..."
result_conf = 0.0

last_time = time.time()

# ==== MAIN LOOP ====
cv2.namedWindow("UsimKD - FP32VINO - Button Mode", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("UsimKD - FP32VINO - Button Mode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Lưu frame vào buffer 5 giây
    buffer_frames.append(frame.copy())

    # Tính FPS hiển thị UI
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now

    # Nếu bấm phím "c" → bắt đầu chấm điểm 5 giây trước
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(buffer_frames) < NUM_FRAMES:
            print("Không đủ dữ liệu trong buffer!")
        else:
            # ==== LẤY 16 FRAMES TỪ MẢNG 5 GIÂY ====
            total = len(buffer_frames)
            idxs = np.linspace(0, total-1, NUM_FRAMES).astype(int)
            clip = [buffer_frames[i] for i in idxs]

            # ==== RUN INFERENCE ====
            x = preprocess(clip)
            logits = compiled(x)[output_layer][0]

            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            idx = int(np.argmax(probs))
            result_label = LABELS[idx]
            result_conf = float(probs[idx])

            # ==== LOG ====
            log_file.write(f"{time.time()},{result_label},{result_conf:.4f}\n")
            log_file.flush()

            print(f"🔥 KẾT QUẢ 5 GIÂY TRƯỚC: {result_label} (Conf={result_conf:.2f})")

    # ==== VẼ UI ====
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    draw.text((20, 40), f"Kết quả: {result_label}", font=font_main, fill=(0,255,0))
    draw.text((20, 90), f"Conf: {result_conf:.2f}", font=font_small, fill=(255,255,0))
    draw.text((20, 135), f"FPS Camera: {fps:.1f}", font=font_small, fill=(255,255,255))
    draw.text((20, 175), f"Bấm 'c' để chấm 5 giây trước", font=font_small, fill=(200,200,200))

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("UsimKD - Button Mode (5s Capture)", frame)

    if key == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
