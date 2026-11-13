#!/bin/bash
# Chạy toàn bộ ablation (Nấc A, +Bước 1, +Bước 2, +Bước 3) trên Kaggle bằng MỘT lệnh.
#   !bash kaggle_run.sh        # 100 câu train, xen kẽ đều giữa các business
#   !bash kaggle_run.sh 30     # chạy thử nhanh 30 câu
#   !bash kaggle_run.sh 100 fix,fix_zero   # chỉ chạy 2 cấu hình sửa lỗi
# Cần: Accelerator = GPU T4 x2, Internet = On, Add Input = dataset chứa thư mục train/ của đề.
cd "$(dirname "$0")"
LIMIT=${1:-100}
ONLY=${2:-}        # ví dụ: fix,fix_zero (bỏ trống = 4 cấu hình ablation cũ)

echo "== 1/4 cài vLLM (nếu lỗi sẽ tự dùng backend hf, chậm hơn)"
pip install -q vllm 2>&1 | tail -2
BACKEND=vllm
python -c "import vllm" 2>/dev/null || { echo "vLLM không cài được -> dùng backend hf"; BACKEND=hf; }
TP=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

echo "== 2/4 tìm dữ liệu train trong /kaggle/input"
TRAIN=$(python - <<'PY'
import glob, os
c = sorted(glob.glob('/kaggle/input/**/train/questions.jsonl', recursive=True))
print(os.path.dirname(c[0]) if c else '')
PY
)
if [ -z "$TRAIN" ]; then
  echo "KHÔNG thấy train/questions.jsonl trong /kaggle/input -> hãy Add Input dataset dữ liệu của đề vào notebook rồi chạy lại."
  exit 1
fi
echo "train = $TRAIN | backend = $BACKEND | số GPU = $TP | số câu = $LIMIT"

echo "== 3/4 chạy ablation (4 cấu hình, nạp model một lần)"
python run_all.py --split_dir "$TRAIN" --out_root /kaggle/working/runs --work_dir /kaggle/working/work \
    --backend $BACKEND --tp $TP --limit $LIMIT --n 4 ${ONLY:+--only $ONLY}

echo "== 4/4 đóng gói kết quả"
python -c "import shutil; print(shutil.make_archive('/kaggle/working/ablation_results', 'zip', '/kaggle/working/runs'))"
echo "XONG. Bảng kết quả ở trên; file zip nằm ở mục Output của notebook."
