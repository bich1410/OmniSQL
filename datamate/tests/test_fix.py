"""Test --hints / --repair bằng dữ liệu giả + backend stub (không cần GPU)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import fix_extra, repair_suffix, select_tables  # noqa: E402
import run_a  # noqa: E402

tmp = Path(tempfile.mkdtemp())
split = tmp / "split"
subprocess.run([sys.executable, str(HERE / "make_mock.py"), str(split)], check=True)
oracle = json.load(open(tmp / "oracle.json", encoding="utf-8"))
# q5: SQL đầu lỗi (cột không tồn tại) -> bản sửa đúng
oracle["q5"] = "SELECT SUM(gia_von) FROM ban_hang"
oracle["q5#repair"] = "SELECT SUM(so_luong) FROM ban_hang"
# q2: SQL đầu trả 0 (lọc sai ngày); chỉ sửa khi bật --repair_zero
oracle["q2"] = "SELECT COUNT(*) FROM ban_hang WHERE ngay = '1 thg 3 2026'"
oracle["q2#repair"] = "SELECT SUM(so_luong) FROM ban_hang WHERE ngay >= '2026-03-01'"
(tmp / "oracle.json").write_text(json.dumps(oracle, ensure_ascii=False), encoding="utf-8")


def run(name, *extra):
    out = tmp / name / "submission.jsonl"
    run_a.main(["--split_dir", str(split), "--out", str(out), "--work_dir", str(tmp / "work"),
                "--backend", "stub", "--oracle", str(tmp / "oracle.json"), "--n", "2", "--threshold", "0.5", *extra])
    logs = {json.loads(l)["id"]: json.loads(l) for l in open(tmp / name / "logs" / "logs.jsonl", encoding="utf-8")}
    return logs


base = run("base")
assert base["q5"]["candidates"][0]["error"], "q5 phải lỗi khi không bật --repair"
assert base["q2"]["answer"]["answer"] == 0

fixed = run("fix", "--hints", "--repair")
c = fixed["q5"]["candidates"][0]
assert c["error"] is None and c["repaired_from"] and c["answer"]["answer"] == 51, c   # đã sửa, ra đáp án
assert fixed["q2"]["answer"]["answer"] == 0                                           # chưa bật repair_zero: giữ 0

fz = run("fixz", "--hints", "--repair", "--repair_zero")
assert fz["q2"]["answer"]["answer"] == 10 and fz["q2"]["candidates"][0]["repair_reason"] == "zero"

# gợi ý giá trị: tên hàng viết không dấu vẫn khớp giá trị thật có dấu
db = tmp / "work" / "biz_001.db"
tables = json.load(open(tmp / "work" / "biz_001.tables.json", encoding="utf-8"))["tables"]
tables = [{"name": t["name"], "cols": [tuple(x) for x in t["cols"]]} for t in tables]
extra = fix_extra(db, tables, "ban duoc bao nhieu son dulux 5l")
assert "Sơn Dulux 5L" in extra and "quý N" in extra, extra
# repair_suffix: chỉ liệt kê bảng có trong SQL
sfx = repair_suffix("SELECT x FROM ban_hang", "error", "no such column: x", tables)
assert "ban_hang(" in sfx and "khach_hang" not in sfx
print("test_fix OK")
