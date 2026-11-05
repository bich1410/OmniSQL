# DataMate – nấc A
Pipeline: câu hỏi → prompt OmniSQL (schema + README + rulebook + as_of_date) → SQL → chạy SQLite → đáp án JSON.
- `common.py`  nạp CSV/XLSX/SQLite vào SQLite, dựng schema/prompt, chạy SQL, đổi kết quả thành {type, answer}
- `run_a.py`   chạy cả split, ghi `submission.jsonl` và `logs/logs.jsonl`
- `score.py`   chấm theo luật của đề (PA, theo kiểu đáp án, theo business)
- `tests/`     dữ liệu giả + test cho bộ chấm và bộ định dạng
- `kaggle_nac_A.ipynb`  notebook chạy trên Kaggle
Chạy thử không cần GPU: `python tests/make_mock.py /tmp/mock/train` rồi `run_a.py --backend stub --oracle ...`
Model khai báo trong README nộp bài: seeklhy/OmniSQL-7B (~7.6B tham số, dưới trần 15B).

## Việc CHƯA chốt — cần làm lại ở nấc B
Dữ liệu thật: rulebook ~6.700 token, nhiều business có 20-28 bảng (mỗi sheet Excel
= 1 bảng) -> schema một mình có thể vượt khung 8192 token của model. Để nấc A chạy
được hôm nay (5/11), đã thêm `select_tables()` và `select_rulebook()` trong
`common.py`: lọc bảng/mục rulebook theo khớp TỪ KHÓA với câu hỏi (string match thô,
không hiểu nghĩa). **Đây chỉ là giải pháp tạm để không nghẽn, CHƯA được đo so với
phương án khác** (lọc theo embedding, để LLM tự chọn bảng cần dùng, viết lại
rulebook rút gọn tay, hoặc tăng context window của model lên quá 8192). Trước khi
tin số điểm nấc A để so sánh với B/C/D/E, nên:
1. Chạy A/B so sánh vài phương án trên tập train (có `answers.jsonl` thật, chấm
   bằng `score.py`) để chọn cách lọc tốt nhất bằng số liệu, không đoán.
2. Xem `logs/logs.jsonl` để biết câu nào bị bỏ sót bảng/luật cần thiết do lọc sai.
