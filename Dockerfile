# Dùng Python phiên bản nhỏ gọn
FROM python:3.10-slim

# Cài đặt biến môi trường để log hiện ra ngay lập tức
ENV PYTHONUNBUFFERED=1

# Tạo thư mục làm việc bên trong máy ảo
WORKDIR /app

# Copy danh sách thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code của bạn vào máy ảo
COPY . .

# Mặc định khi bật máy lên thì chạy file main.py
CMD ["python", "main.py"]