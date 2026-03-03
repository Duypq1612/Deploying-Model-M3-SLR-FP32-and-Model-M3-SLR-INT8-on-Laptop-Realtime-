from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2, numpy as np, openvino as ov, os, time, csv

# ======== CONFIG ========
MODEL_PATH = "UsimKD_student_qat.xml"
LABEL_PATH = "1_199_label.csv"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
NUM_FRAMES = 8
IMAGE_SIZE = 224
CONF_THRESHOLD = 0.1

# ======== LOAD LABELS ========
LABELS = []
with open(LABEL_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "name" in row and row["name"].strip():
            LABELS.append(row["name"].strip())
print(f"✅ Đã load {len(LABELS)} nhãn.")

# ======== LOAD MODEL ========
core = ov.Core()
model = core.read_model(MODEL_PATH)
compiled = core.compile_model(model, "CPU")
output_layer = compiled.output(0)
infer_request = compiled.create_infer_request()
print("✅ Mô hình QAT load thành công.")

# ======== PREPROCESS ========
def preprocess_clip(frames):
    clip = [cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE))[:, :, ::-1] / 255.0 for f in frames]
    x = np.stack(clip).transpose(3, 0, 1, 2)
    x = np.expand_dims(x.astype(np.float32), 0)
    return x

# ======== EXTRACT CLIP FROM VIDEO ========
def extract_frames(path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return []
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(frame)
    cap.release()
    return frames

# ======== FASTAPI SETUP ========
app = FastAPI(title="UsimKD QAT Video Upload Inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    # Lưu file tạm
    save_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(save_path, "wb") as f:
        f.write(await video.read())

    # Xử lý video
    frames = extract_frames(save_path, NUM_FRAMES)
    if len(frames) == 0:
        return {"error": "Video không hợp lệ hoặc rỗng"}

    x = preprocess_clip(frames)
    result = infer_request.infer({0: x})
    res = result[output_layer]
    conf = float(np.max(res))
    idx = int(np.argmax(res))
    label = LABELS[idx] if idx < len(LABELS) else f"Từ {idx}"
    top5_idx = np.argsort(res[0])[::-1][:5]
    top5 = [{"rank": i+1,
         "label": LABELS[k] if k < len(LABELS) else f"Từ {k}",
         "conf": float(res[0][k])} for i, k in enumerate(top5_idx)]
    return {
        "label_vi": label,
        "confidence": round(conf, 3),
        "frames_used": len(frames),
        "video_name": video.filename,
        "top5": top5
}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)
