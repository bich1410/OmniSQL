import sys; sys.path.insert(0, "..")
from score import score_one, norm_str
from common import to_answer, extract_sql

def S(pred, ref): return score_one(pred, ref)[0]
# string: chuẩn hoá, alias, giữ dấu
assert S({"type":"string","answer":"  STING dâu 330ml! "},{"type":"string","answer":"Sting dâu 330ml"})==1
assert S({"type":"string","answer":"sp0042"},{"type":"string","answer":"Sting dâu 330ml","alias":["SP0042"]})==1
assert S({"type":"string","answer":"Sting dau 330ml"},{"type":"string","answer":"Sting dâu 330ml"})==-1   # dấu không được bỏ
# number / percent
assert S({"type":"number","answer":48250000.0},{"type":"number","answer":48250000})==1
assert S({"type":"percent","answer":-12.454},{"type":"percent","answer":-12.45})==1
assert S({"type":"percent","answer":12.45},{"type":"percent","answer":-12.45})==-1                      # sai dấu
assert S({"type":"number","answer":5},{"type":"percent","answer":5})==-1                                  # sai kiểu
# list: có thứ tự
assert S({"type":"list","answer":["a","b"]},{"type":"list","answer":["a","b"]})==1
assert S({"type":"list","answer":["b","a"]},{"type":"list","answer":["a","b"]})==-1
# table: cột đổi chỗ vẫn khớp theo tên, hàng là tập
t1={"type":"table","answer":{"columns":["khach_hang","so_tien_no"],"rows":[["Chị Lan",3200000],["Anh Tú",1500000]]}}
t2={"type":"table","answer":{"columns":["so_tien_no","khach_hang"],"rows":[[1500000,"Anh Tú"],[3200000,"Chị Lan"]]}}
assert S(t2,t1)==1
# unanswerable / abstain / thiếu
U={"type":"unanswerable","answer":None}
assert S({"type":"unanswerable"},U)==1
assert S({"type":"number","answer":1},U)==-1          # trả lời câu không trả lời được = sai
assert S({"type":"abstain"},U)==0 and S(None,U)==0
assert S({"type":"unanswerable"},{"type":"number","answer":1})==-1
# to_answer
assert to_answer("tăng bao nhiêu phần trăm",["x"],[(12.456,)])=={"type":"percent","answer":12.46}
assert to_answer("còn hàng không",["x"],[(1,)])=={"type":"boolean","answer":True}
assert to_answer("ngày nào bán nhiều nhất",["d"],[("15/08/2026",)])=={"type":"date","answer":"2026-08-15"}
assert to_answer("x",["c"],[])=={"type":"abstain"}
assert to_answer("x",["c"],[(None,)])=={"type":"abstain"}
assert to_answer("tổng",["c"],[(5.0,)])=={"type":"number","answer":5}
assert extract_sql("blah\n```sql\nSELECT 1\n```\nxong")=="SELECT 1"
assert extract_sql("không có khối code") is None
print("tất cả test đạt")
