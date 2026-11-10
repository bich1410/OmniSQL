import sys; sys.path.insert(0, "..")
from vote import answer_key, apply_threshold, vote

A = {"type": "abstain"}
def num(x): return {"type": "number", "answer": x}
def s(x): return {"type": "string", "answer": x}

# cùng đáp án dù khác cách viết: số làm tròn 2 chữ số, chuỗi chuẩn hóa như score.py (giữ dấu)
assert answer_key(num(5)) == answer_key(num(5.001))
assert answer_key(s("Sting dâu 330ml")) == answer_key(s(" STING dâu 330ml! "))
assert answer_key(s("Sting dau")) != answer_key(s("Sting dâu"))          # dấu không được bỏ
assert answer_key(num(5)) != answer_key({"type": "percent", "answer": 5})  # khác kiểu = khác đáp án
# list có thứ tự; table: đổi chỗ cột/hàng vẫn cùng đáp án
assert answer_key({"type": "list", "answer": ["a", "b"]}) != answer_key({"type": "list", "answer": ["b", "a"]})
t1 = {"type": "table", "answer": {"columns": ["khach", "no"], "rows": [["Lan", 3], ["Tu", 1]]}}
t2 = {"type": "table", "answer": {"columns": ["no", "khach"], "rows": [[1, "Tu"], [3, "Lan"]]}}
assert answer_key(t1) == answer_key(t2)

# 3/4 đồng ý -> thắng với độ đồng thuận 0.75
ans, conf, idx = vote([num(10), num(10), num(7), num(10)])
assert ans == num(10) and conf == 0.75 and idx == 0
# lỗi/rỗng không là phiếu nhưng vẫn nằm trong mẫu số: 2 phiếu / 4 candidate = 0.5
ans, conf, idx = vote([A, num(10), A, num(10)])
assert ans == num(10) and conf == 0.5 and idx == 1
# toàn bộ abstain -> abstain, đồng thuận 0
assert vote([A, A, A]) == (A, 0.0, None)
# hòa 2-2: ưu tiên nhóm có candidate greedy (chỉ số 0)
ans, conf, idx = vote([num(1), num(2), num(2), num(1)])
assert ans == num(1) and conf == 0.5 and idx == 0
# hòa mà greedy lỗi: nhóm xuất hiện sớm hơn thắng
ans, _, idx = vote([A, num(2), num(1), num(1), num(2)])
assert ans == num(2) and idx == 1
# n=1 giống nấc A: một đáp án -> conf 1.0; một lỗi -> abstain
assert vote([num(3)]) == (num(3), 1.0, 0) and vote([A]) == (A, 0.0, None)

# ngưỡng: >= ngưỡng thì trả lời, còn lại abstain
assert apply_threshold(num(10), 0.5, 0.5) == num(10)
assert apply_threshold(num(10), 0.25, 0.5) == A
print("test_vote đạt")