# Nhật ký thực hiện

Ngày: 02/11/2026 (Asia/Saigon).

## Yêu cầu và dữ liệu

- Đọc `yeucau.txt`: phân loại chủ đề AG News bằng BERT hoặc DistilBERT;
  so sánh fine-tuning với frozen encoder.
- Kiểm tra dữ liệu local: `datasets/train.csv` có 120.000 mẫu (30.000 mỗi lớp),
  `datasets/test.csv` có 7.600 mẫu (1.900 mỗi lớp).
- Xác nhận CSV có ba cột `Class Index`, `Title`, `Description`.
- Kiểm tra Git: thư mục đã có `.git`, chưa có remote; không thực hiện commit/push.

## Các file đã tạo

- `ag_news_classification.ipynb`: notebook 16 cell, giải thích bằng tiếng Việt.
  Đọc/kiểm tra CSV, chuẩn hóa văn bản, loại trùng và nhãn mâu thuẫn trong train,
  tách validation phân tầng, tokenization và padding động.
- Hai cấu hình DistilBERT: đóng băng backbone và fine-tuning toàn bộ.
  Dùng cùng split/seed; learning rate riêng cho mỗi cấu hình, giải thích trong notebook.
- Lưu checkpoint tốt nhất theo validation macro-F1; đánh giá test bằng Accuracy,
  macro-F1, báo cáo từng lớp, confusion matrix; lưu biểu đồ, CSV và cấu hình chạy.
- Bổ sung nạp lại checkpoint và dự đoán tin tiếng Anh mới.
- `requirements.txt`: thư viện cần cài.
- `.gitignore`: bỏ qua dữ liệu CSV, checkpoint, môi trường ảo, cache và file môi trường.
- `README.md`: hướng dẫn cài đặt, chạy thử/chạy đủ và tạo repo GitHub bằng PowerShell.
- `build_notebook.py`: tái tạo notebook từ mã nguồn.
- `verify_notebook.py`: kiểm tra tích hợp offline bằng DistilBERT cực nhỏ khởi tạo ngẫu nhiên.

## Kiểm tra và giới hạn

- Sinh notebook thành công; kiểm tra schema bằng `nbformat.validate` và cú pháp
  tất cả code cell bằng `ast.parse` đạt.
- Môi trường hiện có Python 3.10.11, PyTorch 2.14.1+cpu, Transformers 5.20.2;
  không có CUDA.
- Lần chạy kiểm tra tích hợp đầu tiên bị lỗi quyền ghi safetensors tại thư mục tạm;
  đã yêu cầu chạy lại với quyền phù hợp. Kết quả cuối được ghi bổ sung bên dưới.
- Chưa chạy huấn luyện pretrained trên toàn bộ AG News. Không công bố Accuracy/F1
  thực nghiệm khi chưa huấn luyện. Notebook mặc định chạy thử 2.000/400/400 mẫu,
  1 epoch; đổi `QUICK_RUN=False` để chạy đủ với 3 epoch.
- Không tự tạo repo trên tài khoản GitHub; người dùng được hướng dẫn thực hiện trong README.

### Kết quả kiểm tra cuối

- Chạy `python -X utf8 verify_notebook.py`: đạt toàn bộ kiểm tra tích hợp offline.
  Chạy mọi code cell với mô hình DistilBERT nhỏ ngẫu nhiên và 8 mẫu mỗi split.
- Xác nhận backbone frozen không đổi trọng số sau optimizer step, head có cập nhật,
  và fine-tuning bật gradient cho toàn bộ tham số.
- Cả hai vòng huấn luyện, validation, lưu/nạp checkpoint, báo cáo test, biểu đồ,
  dự đoán và từ chối đầu vào rỗng đều chạy thành công.
- Kiểm tra dữ liệu phát hiện 83 bản ghi trùng văn bản trong train, 15 nhóm văn bản
  có nhãn mâu thuẫn; 4 bản ghi test có văn bản xuất hiện trong pool train đã làm sạch.
  Notebook báo cáo giao nhau này, giữ nguyên test và không dùng test để chọn mô hình.
- Khắc phục vấn đề encoding terminal bằng UTF-8; script kiểm tra cũng tự cấu hình stdout UTF-8.
- Điểm số từ mô hình ngẫu nhiên chỉ phục vụ kiểm tra kỹ thuật, không được đưa vào notebook
  như kết quả AG News. Checkpoint kiểm tra nằm trong thư mục tạm và tự được dọn sau khi chạy.

## Nguồn đối chiếu

- https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
- https://huggingface.co/datasets/fancyzhx/ag_news
- https://huggingface.co/docs/transformers/model_doc/distilbert
- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
