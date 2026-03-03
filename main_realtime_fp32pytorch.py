import cv2
import numpy as np
import time
import csv, os, sys
from collections import deque
from PIL import ImageFont, ImageDraw, Image

import torch
import torch.nn.functional as F
import yaml
from modelling.Uniformer import UsimKD
from utils.utils import load_model


# ====================== CONFIG ======================
CKPT_PATH = "models/best_checkpoints1.pth"
ROOT = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(ROOT, "configs", "UsimKD_MultiVSL200.yaml")
LABEL_PATH = "1_199_label.csv"

NUM_FRAMES = 8
IMAGE_SIZE = 224
CAPTURE_SECONDS = 5
CAMERA_FPS = 30
BUFFER_SIZE = CAPTURE_SECONDS * CAMERA_FPS

LOG_PATH = "fp32_pytorch_button_log.txt"
WINDOW_NAME = "UsimKD - FP32 PyTorch - Button Mode"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)


# ====================== BUILD MODEL ======================
def build_model_pytorch():
    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["num_classes"] = len(LABELS)
    cfg["training"]["pretrained"] = False
    cfg["training"]["pretrained_model"] = None
    cfg["training"]["device"] = str(DEVICE)

    model = load_model({
        "data": cfg["data"],
        "model": cfg["model"],
        "training": cfg["training"]
    })

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    print("PyTorch FP32 model loaded.")
    return model


print("Đang load mô hình PyTorch FP32...")
model = build_model_pytorch()


# ====================== CAMERA ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera.")
    sys.exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera đã sẵn sàng. Nhấn phím 'c' để nhận diện 5 giây trước.")

# FULLSCREEN
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ====================== BUFFER ======================
buffer_frames = deque(maxlen=BUFFER_SIZE)

# ====================== FONT ======================
def get_font(size=32):
    paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
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

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        buffer_frames.append(frame.copy())

        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("Đang nhận diện 5 giây trước bằng PyTorch FP32...")

            total = len(buffer_frames)
            if total < NUM_FRAMES:
                print("Không đủ dữ liệu.")
            else:
                idxs = np.linspace(0, total-1, NUM_FRAMES).astype(int)
                clip = [buffer_frames[i] for i in idxs]

                x = preprocess(clip)
                print("\n=== DEBUG INPUT ===")
                print("x:", type(x), x.shape if isinstance(x, torch.Tensor) else None)
                print("===================\n")

                logits_out = model(
                        rgb_left=None,
                        rgb_center=x,
                        rgb_right=None
                    )
                logits = logits_out["logits"]


                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                idx = int(np.argmax(probs))

                result_label = LABELS[idx]
                result_conf = float(probs[idx])

                print(f"Kết quả: {result_label} | Độ tin cậy={result_conf:.2f}")

                log_f.write(f"{time.time()},{result_label},{result_conf:.4f}\n")
                log_f.flush()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img_pil)

        draw.text((20, 40), f"Kết quả: {result_label}", font=font_main, fill=(0,255,0))
        draw.text((20, 90), f"Độ tin cậy: {result_conf:.2f}", font=font_small, fill=(255,255,0))
        draw.text((20,140), f"FPS Camera: {fps:.1f}", font=font_small, fill=(255,255,255))
        draw.text((20,180), "Nhấn 'c' để nhận diện 5 giây trước", font=font_small, fill=(200,200,200))

        frame_show = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, frame_show)

        if key == ord('q'):
            break

cap.release()
log_f.close()
cv2.destroyAllWindows()
