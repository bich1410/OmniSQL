"""Tạo split GIẢ đúng cấu trúc đề, để kiểm thử pipeline (không phải dữ liệu thật)."""
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]); biz = root / "data" / "biz_001"; biz.mkdir(parents=True, exist_ok=True)
sales = pd.DataFrame({
    "Ngày": ["2026-02-23","2026-02-24","2026-02-25","2026-02-26","2026-03-02","2026-03-03"],
    "Tên SP": ["Xi măng Hà Tiên PCB40 bao 50kg","Sơn Dulux 5L","Xi măng Hà Tiên PCB40 bao 50kg","Gạch men 60x60","Sơn Dulux 5L","Gạch men 60x60"],
    "Số lượng": [10, 3, 8, 20, 4, 6],
    "Đơn giá": [95000, 1200000, 95000, 180000, 1200000, 180000],
})
sales.to_csv(biz / "ban_hang.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(biz / "khach_hang.xlsx") as w:
    pd.DataFrame({"Khách": ["Chị Lan","Anh Tú"], "Số tiền nợ": [3200000, 1500000]}).to_excel(w, sheet_name="Công nợ", index=False)
    pd.DataFrame({"Khách": ["Chị Lan","Anh Tú"], "SĐT": ["0900","0911"]}).to_excel(w, sheet_name="Liên hệ", index=False)
(biz / "README.md").write_text("Cửa hàng vật liệu xây dựng. ban_hang: mỗi dòng là một lần bán.", encoding="utf-8")
(root / "rulebook.md").write_text("bán chạy = theo số lượng. Tuần chạy từ thứ Hai đến Chủ nhật.", encoding="utf-8")
(root / "calendar.csv").write_text("solar,lunar\n2026-02-17,01-01\n", encoding="utf-8")

Q = [
 ("q1","sp bán dc nhiều nhất tuần rồi là gì","2026-03-04","string","Gạch men 60x60",
  "SELECT ten_sp FROM ban_hang WHERE ngay BETWEEN '2026-02-23' AND '2026-03-01' GROUP BY ten_sp ORDER BY SUM(so_luong) DESC LIMIT 1"),
 ("q2","tổng số lượng bán được trong tháng 3","2026-03-04","number",10,
  "SELECT SUM(so_luong) FROM ban_hang WHERE ngay >= '2026-03-01'"),
 ("q3","3 mặt hàng bán chạy nhất","2026-03-04","list",["Gạch men 60x60","Xi măng Hà Tiên PCB40 bao 50kg","Sơn Dulux 5L"],
  "SELECT ten_sp FROM ban_hang GROUP BY ten_sp ORDER BY SUM(so_luong) DESC, ten_sp LIMIT 3"),
 ("q4","ai đang nợ tiền, bao nhiêu","2026-03-04","table",{"columns":["khach","so_tien_no"],"rows":[["Chị Lan",3200000],["Anh Tú",1500000]]},
  "SELECT khach, so_tien_no FROM khach_hang_cong_no"),
 ("q5","lợi nhuận tháng này là bao nhiêu","2026-03-04","unanswerable",None,
  "SELECT SUM(gia_von) FROM ban_hang"),                       # cột không tồn tại -> SQL lỗi -> abstain
 ("q6","ngày nào bán nhiều nhất","2026-03-04","date","2026-02-26",
  "SELECT ngay FROM ban_hang GROUP BY ngay ORDER BY SUM(so_luong) DESC LIMIT 1"),
 ("q7","doanh thu tuần rồi bao nhiêu","2026-03-04","number",999,   # cố tình SQL sai -> phải bị chấm sai
  "SELECT SUM(so_luong*don_gia) FROM ban_hang WHERE ngay BETWEEN '2026-02-23' AND '2026-03-01'"),
]
qs, ans, oracle = [], [], {}
for qid,q,asof,t,a,sql in Q:
    qs.append({"id":qid,"business_id":"biz_001","question":q,"as_of_date":asof})
    ans.append({"id":qid,"type":t,"answer":a, **({"alias":["SP0001"]} if t=="string" else {})})
    oracle[qid]=sql
for name,rows in (("questions.jsonl",qs),("answers.jsonl",ans)):
    (root/name).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
(root.parent/"oracle.json").write_text(json.dumps(oracle,ensure_ascii=False),encoding="utf-8")
