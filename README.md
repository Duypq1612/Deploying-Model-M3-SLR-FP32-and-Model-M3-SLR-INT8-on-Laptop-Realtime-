# 🤟 AI Nhận Dạng Ngôn Ngữ Ký Hiệu (Sign Language Recognition)

Chào mừng bạn đến với dự án AI nhận dạng ngôn ngữ ký hiệu! Đây là một hệ thống được thiết kế để chạy mượt mà trên thiết bị laptop cá nhân. Dự án hỗ trợ nhận diện qua luồng video trực tiếp (webcam realtime) và cung cấp một API server (FastAPI) để xử lý các video được tải lên.

![Demo]([Thêm link ảnh hoặc ảnh GIF bạn đang test nhận dạng thành công vào đây])

## ✨ Tính năng nổi bật

* **⚡ Nhận dạng Realtime qua Webcam:** Xử lý trực tiếp hình ảnh từ camera laptop, hiển thị khung hình và kết quả từ vựng ngôn ngữ ký hiệu ngay trên màn hình.
* **🎯 Độ chính xác ổn định:** Mô hình đạt độ chính xác lên đến 78% trong điều kiện thực tế.
* **🧠 Đa dạng phiên bản Model:** Tích hợp script chạy thử nghiệm qua lại giữa 4 phiên bản model để so sánh hiệu năng:
    * Mô hình gốc (Checkpoint FP32)
    * Mô hình ONNX (FP32)
    * Mô hình lượng tử hóa (Quantization INT8) - 2 phiên bản
* **🌐 Web API chạy ngầm (FastAPI):** Cung cấp API endpoint `/predict` xử lý video tải lên, tự động trích xuất 8 khung hình và sử dụng mô hình tối ưu siêu nhẹ (OpenVINO QAT) để dự đoán top 5 kết quả chính xác nhất.

## 📂 Cấu trúc thư mục

\`\`\`text
nhan-dang-ngon-ngu-ky-hieu/
│
├── models/                  # Chứa file nhãn .csv và text hướng dẫn tải model (Vì file model lớn nên không tải trực tiếp lên GitHub)
├── main_realtime_(..).py            # Chứa script chạy nhận dạng qua Webcam laptop
│       
├── script_check.py              # Chứa mã nguồn Web API
│            
│── uploads/                 # Thư mục tạm lưu video (tự động dọn dẹp)
├── requirements.txt         # Danh sách các thư viện cần cài đặt
└── README.md                # Tài liệu hướng dẫn (Bạn đang đọc nó đây!)
\`\`\`

## 🚀 Hướng dẫn Cài đặt

**Bước 1:** Clone (Tải) kho lưu trữ này về máy tính của bạn:
\`\`\`bash
git clone 
cd UsimKD
\`\`\`

**Bước 2:** Cài đặt các thư viện cần thiết:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**Bước 3:** Tải Model AI
* Vì giới hạn dung lượng của GitHub, các file model không được tải lên trực tiếp.
* Vui lòng tải 4 file model tại [https://drive.google.com/drive/folders/12Y0e17dUGkOCPnOiLAkddmU5CMDqNegb?usp=sharing] 

## 💻 Hướng dẫn Sử dụng

### 1. Chạy nhận dạng trực tiếp qua Webcam (Realtime)
Mở terminal và chạy lệnh sau để bật camera:
\`\`\`bash
python main_realtime_(..).py
\`\`\`
*(Nhấn phím `q` để thoát camera)*

### 2. Chạy Web API Server (FastAPI)
Mở terminal và khởi động server:
\`\`\`bash
python script_check.py
\`\`\`
* Server sẽ chạy tại địa chỉ: `http://localhost:8050`
* Bạn có thể xem tài liệu API tự động sinh ra và test trực tiếp trên trình duyệt bằng cách truy cập: `http://localhost:8050/docs`

## 🤝 Ghi chú của tác giả
Dự án này được xây dựng trong quá trình tôi bắt đầu hành trình tự học lập trình và AI từ những bước đầu tiên. Rất mong nhận được sự góp ý từ mọi người!
