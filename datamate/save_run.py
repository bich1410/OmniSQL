"""Đóng gói MỘT lần chạy để lưu lại và so sánh với các bản cải tiến sau này.

  python save_run.py --run_dir /kaggle/working/runs/base_n4 \
                     --ref {DATA}/train/answers.jsonl --questions {DATA}/train/questions.jsonl

run_dir phải có: submission.jsonl, logs/logs.jsonl, logs/config.json (do run_a.py ghi).
Tạo thêm trong run_dir:
  pred_greedy.jsonl  đáp án của candidate greedy (= SQL đầu tiên, chính là kết quả "không bỏ phiếu")
  score_greedy.txt   điểm của bản greedy
  score_vote.txt     điểm sau khi bỏ phiếu + abstain (đúng như submission.jsonl)
  calibrate.txt      bảng quét ngưỡng
  meta.json          cấu hình, commit git, ngày giờ, PA hai bản
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sh(args):
    r = subprocess.run(args, capture_output=True, text=True, cwd=HERE)
    return (r.stdout + r.stderr).strip()


def pa_of(text):
    m = re.search(r"PA=([+-]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def cov_acc(text):
    m = re.search(r"độ phủ[^=]*= ([0-9.]+) \| độ chính xác[^=]*= ([0-9.]+)", text)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def save(run_dir, ref, questions, quiet=False):
    run = Path(run_dir)
    logs = [json.loads(l) for l in open(run / "logs" / "logs.jsonl", encoding="utf-8") if l.strip()]
    cfg = json.load(open(run / "logs" / "config.json", encoding="utf-8"))

    with open(run / "pred_greedy.jsonl", "w", encoding="utf-8") as f:   # candidate 0 = greedy
        for l in logs:
            ans = l["candidates"][0]["answer"] if l["candidates"] else {"type": "abstain"}
            f.write(json.dumps({"id": l["id"], **ans}, ensure_ascii=False) + "\n")

    score = [sys.executable, "score.py", "--ref", str(ref), "--questions", str(questions), "--only_run", "--pred"]
    s_greedy = sh(score + [str(run / "pred_greedy.jsonl")])
    s_vote = sh(score + [str(run / "submission.jsonl")])
    calib = sh([sys.executable, "calibrate.py", "--logs", str(run / "logs" / "logs.jsonl"),
                "--ref", str(ref), "--questions", str(questions)])
    (run / "score_greedy.txt").write_text(s_greedy + "\n", encoding="utf-8")
    (run / "score_vote.txt").write_text(s_vote + "\n", encoding="utf-8")
    (run / "calibrate.txt").write_text(calib + "\n", encoding="utf-8")

    try:                                              # chẩn đoán từng câu (đáp án chuẩn vs dự đoán)
        import diagnose
        diagnose.write(run, ref, questions)
    except Exception as e:
        print("diagnose lỗi:", e)

    commit = sh(["git", "rev-parse", "--short", "HEAD"])
    cov, acc = cov_acc(s_vote)
    meta = {"run": run.name, "config": cfg, "n_questions": len(logs),
            "git_commit": commit if re.fullmatch(r"[0-9a-f]{7,40}", commit) else None,
            "PA_greedy": pa_of(s_greedy), "PA_vote": pa_of(s_vote),
            "vote_coverage": cov, "vote_accuracy": acc, "seconds": cfg.get("seconds")}
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not quiet:
        print("=== greedy (không bỏ phiếu) ===\n" + "\n".join(s_greedy.splitlines()[:3]))
        print("\n=== bỏ phiếu + abstain ===\n" + "\n".join(s_vote.splitlines()[:3]))
        print("\n=== quét ngưỡng ===\n" + calib)
        print(f"\nĐã lưu vào {run}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--questions", required=True)
    a = ap.parse_args()
    save(a.run_dir, a.ref, a.questions)


if __name__ == "__main__":
    main()
