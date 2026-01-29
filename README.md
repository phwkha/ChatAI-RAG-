Để chạy hệ thống lần đầu tiên (khi mới tải code về hoặc mới cài lại máy)

BƯỚC 1: Khởi tạo và Bật máy
  Mở Terminal tại thư mục dự án và chạy lệnh này để Docker tự động cài đặt mọi thứ:
    docker compose up -d --build (Chờ vài phút để nó tải Python và dựng máy ảo).

BƯỚC 2: Tải "Bộ não" cho AI (BẮT BUỘC)
  Lần đầu tiên chạy, máy AI chưa có dữ liệu trí tuệ. Bạn phải chạy 2 lệnh này để tải về (chỉ cần làm 1 lần duy nhất trong đời):
    Tải model tư duy (DeepSeek):
      docker exec -it ollama ollama pull deepseek-r1:1.5b
    Tải model đọc hiểu văn bản (Nomic):
      docker exec -it ollama ollama pull nomic-embed-text

BƯỚC 3: Vào Chat
  Sau khi tải xong ở bước 2, bạn chạy lệnh này để vào giao diện chat:
    docker attach app
  ⚠️ Lưu ý quan trọng: Sau khi gõ lệnh ở Bước 3, nếu thấy màn hình đen thui hoặc đứng im, hãy BẤM PHÍM ENTER một cái. Dòng chữ 🗣️ Bạn hỏi: sẽ hiện ra ngay lập tức!


Để sử dụng (bật khi cần và tắt khi xong để tiết kiệm điện/RAM), bạn chỉ cần nhớ đúng 2 bộ lệnh này thôi:

1. KHI CẦN DÙNG (BẬT MÁY)
  Mở Terminal tại thư mục code và gõ lần lượt:

  Khởi động hệ thống:
    docker compose up -d
    
  Vào màn hình chat:
    docker attach app
(Nếu thấy màn hình đen thui, nhớ bấm phím Enter một cái để đánh thức nó nhé!)

2. KHI KHÔNG DÙNG (TẮT MÁY)
  Khi chat xong, bạn làm như sau để tắt sạch sẽ:

    Thoát ra: Gõ exit hoặc thoat rồi Enter.

    Tắt hẳn (Quan trọng): Chạy lệnh này để giải phóng RAM cho máy tính:

      docker compose down
   (Thấy chữ "Removed" hiện ra là xong, máy bạn đã nhẹ tênh).

TÓM TẮT CHO NHANH (Mẹo Copy-Paste)
Lần sau bạn cứ copy dòng này dán vào là xong:

Bật: docker compose up -d && docker attach app

Tắt: docker compose down


Để nâng cấp lên mô hình mạnh hơn (ví dụ từ phiên bản 1.5b lên 8b để AI thông minh hơn, logic tốt hơn), bạn cần thực hiện đúng 3 bước sau.

(Lưu ý: Hướng dẫn này dành cho máy có Card màn hình rời như NVIDIA RTX 3060,.. , vì chạy bằng CPU sẽ rất chậm).

BƯỚC 1: Tải "Bộ não" mới về máy
Bạn cần lệnh cho máy AI tải phiên bản 8 tỷ tham số về. Mở Terminal và chạy:

docker exec -it ollama ollama pull deepseek-r1:8b
(Chờ tải khoảng 4.5GB).

BƯỚC 2: Sửa Code để nhận não mới
Bạn mở file main.py và sửa dòng chọn model (khoảng dòng 44):

Cũ: model="deepseek-r1:1.5b",

Mới: model="deepseek-r1:8b",

BƯỚC 3: Bật GPU trong Docker (Quan Trọng Nhất) 🚀
Để mô hình mạnh chạy mượt, bạn bắt buộc phải cho Docker dùng Card đồ họa. Hãy mở file docker-compose.yml và thêm đoạn mã deploy vào dưới phần ollama-service.

File docker-compose.yml của bạn sẽ trông như thế này sau khi sửa:

YAML
services:
  ollama-service:
    image: ollama/ollama:latest
    container_name: ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    # --- THÊM ĐOẠN NÀY ĐỂ KÍCH HOẠT GPU ---
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # --------------------------------------

  python-app:
    # ... (giữ nguyên phần bên dưới)
BƯỚC 4: Áp dụng thay đổi
Sau khi sửa xong, bạn chạy lệnh này để tái tạo lại hệ thống với cấu hình mới:

docker compose up -d --force-recreate
