"""Chạy toàn bộ ablation MỘT lần (nạp model một lần), lưu từng cấu hình và in bảng so sánh.

  python run_all.py --split_dir /kaggle/input/<ds>/train --out_root /kaggle/working/runs \
                    --backend vllm --tp 2 --limit 100 --n 4

4 cấu hình (mỗi cấu hình sinh n SQL: SQL đầu tiên là greedy = "không bỏ phiếu", cả n SQL = bỏ phiếu):
  base         Nấc A + Bước 1
  expand       + Bước 2 (bung viết tắt)
  link         + Bước 3 (chọn bảng bằng embedding)
  expand_link  + cả hai
  fix, fix_zero  sửa theo phân tích lỗi: --hints + --repair (fix_zero: cả kết quả 0)
Một cấu hình lỗi không làm hỏng các cấu hình còn lại. Kết quả mỗi cấu hình nằm ở <out_root>/<tên>_n<n>_<limit>/.
"""
import argparse
import csv
import json
import traceback
from pathlib import Path

import run_a
import save_run
from abbrev import list_uncovered
from common import build_db  # noqa: F401  (để lỗi import báo sớm)

CONFIGS = {
    "base": [],
    "expand": ["--expand"],
    "link": ["--link"],
    "expand_link": ["--expand", "--link"],
    "fix": ["--hints", "--repair"],                          # gợi ý giá trị ô + sửa SQL lỗi/rỗng
    "fix_zero": ["--hints", "--repair", "--repair_zero"],    # như trên, kết quả 0 cũng sinh lại
}


def headers_of(split_dir):
    """Tiêu đề cột gốc của mọi file CSV trong split (đọc dòng đầu), để báo cáo chữ viết tắt chưa phủ."""
    hs = []
    for p in (Path(split_dir) / "data").glob("*/*.csv"):
        try:
            hs += open(p, encoding="utf-8-sig").readline().strip().split(",")
        except Exception:
            pass
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="thư mục train (có questions.jsonl, answers.jsonl, data/)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--work_dir", default="/tmp/datamate_work")
    ap.add_argument("--backend", choices=["vllm", "hf", "stub"], default="vllm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--embedder", choices=["bge", "hash"], default="bge")
    ap.add_argument("--only", default="", help="chỉ chạy một phần cấu hình, ví dụ: base,expand")
    ap.add_argument("--oracle", default=None, help="(stub)")
    a = ap.parse_args()

    split = Path(a.split_dir)
    ref, qs = split / "answers.jsonl", split / "questions.jsonl"
    out_root = Path(a.out_root)
    default = ["base", "expand", "link", "expand_link"]     # fix/fix_zero chỉ chạy khi chọn bằng --only
    names = [c for c in CONFIGS if (c in a.only.split(",") if a.only else c in default)]

    unc = list_uncovered(headers_of(split))
    print("Chữ viết tắt trong tên cột CHƯA có trong từ điển (token, số lần):", unc[:25] or "không có")

    rows = []
    for name in names:
        run_name = f"{name}_n{a.n}_{'train' + str(a.limit) if a.limit else 'trainall'}"
        run_dir = out_root / run_name
        argv = ["--split_dir", str(split), "--out", str(run_dir / "submission.jsonl"),
                "--work_dir", a.work_dir, "--backend", a.backend, "--tp", str(a.tp),
                "--batch_size", str(a.batch_size), "--n", str(a.n), "--embedder", a.embedder,
                "--spread", "--reuse_db"] + (["--limit", str(a.limit)] if a.limit else []) + CONFIGS[name]
        if a.oracle:
            argv += ["--oracle", a.oracle]
        print(f"\n######## {run_name} ########", flush=True)
        try:
            run_a.main(argv)
            meta = save_run.save(run_dir, ref, qs, quiet=True)
            rows.append({"config": name, **{k: meta[k] for k in
                         ("n_questions", "PA_greedy", "PA_vote", "vote_coverage", "vote_accuracy", "seconds")}})
        except Exception:
            err = traceback.format_exc()
            print("LỖI cấu hình", name, "\n", err, flush=True)
            rows.append({"config": name, "error": err.strip().splitlines()[-1]})

    out_root.mkdir(parents=True, exist_ok=True)
    cols = ["config", "n_questions", "PA_greedy", "PA_vote", "vote_coverage", "vote_accuracy", "seconds", "error"]
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda x: "-" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))   # PA có dấu
    pct = lambda x: "-" if x is None else f"{x:.3f}"
    lines = ["| cấu hình | N | PA greedy | PA bỏ phiếu | độ phủ | chính xác khi trả lời | giây |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['config']} | LỖI: {r['error'][:80]} | | | | | |")
        else:
            lines.append(f"| {r['config']} | {r['n_questions']} | {fmt(r['PA_greedy'])} | {fmt(r['PA_vote'])} | "
                         f"{pct(r['vote_coverage'])} | {pct(r['vote_accuracy'])} | {r['seconds']} |")
    table = "\n".join(lines)
    (out_root / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n================ KẾT QUẢ ================\n" + table)
    print("\nLưu ý: PA bỏ phiếu dùng ngưỡng 0.5 (chưa hiệu chỉnh). Với ~100 câu, chênh lệch dưới ~0.1 chưa đủ kết luận.")


if __name__ == "__main__":
    main()
