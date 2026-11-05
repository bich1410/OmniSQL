# DataMate – nấc A
Pipeline: câu hỏi → prompt OmniSQL (schema + README + rulebook + as_of_date) → SQL → chạy SQLite → đáp án JSON.
- `common.py`  nạp CSV/XLSX/SQLite vào SQLite, dựng schema/prompt, chạy SQL, đổi kết quả thành {type, answer}
- `run_a.py`   chạy cả split, ghi `submission.jsonl` và `logs/logs.jsonl`
- `score.py`   chấm theo luật của đề (PA, theo kiểu đáp án, theo business)
- `tests/`     dữ liệu giả + test cho bộ chấm và bộ định dạng
- `kaggle_nac_A.ipynb`  notebook chạy trên Kaggle
Chạy thử không cần GPU: `python tests/make_mock.py /tmp/mock/train` rồi `run_a.py --backend stub --oracle ...`
Model khai báo trong README nộp bài: seeklhy/OmniSQL-7B (~7.6B tham số, dưới trần 15B).
