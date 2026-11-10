"""Bước 1: bỏ phiếu (self-consistency) giữa nhiều SQL candidate.

Ý tưởng: cho model sinh n SQL (1 greedy + n-1 lấy mẫu), chạy hết, rồi gom các
candidate cho CÙNG một đáp án thành một nhóm. Nhóm đông nhất thắng; độ đồng thuận
(số phiếu của nhóm thắng / n) dùng để quyết định trả lời hay abstain.
"""
import json

from score import norm_str


def _cell(c):
    """Một ô trong list/table -> khóa so sánh (cùng cách chuẩn hóa với score.py)."""
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return ("n", round(float(c), 2))
    return ("s", norm_str(c))


def answer_key(ans: dict):
    """Khóa đại diện cho 'cùng một đáp án' (khớp với cách score.py coi là bằng nhau)."""
    t, a = ans.get("type"), ans.get("answer")
    if t in ("number", "percent"):
        return (t, round(float(a), 2))
    if t == "string":
        return (t, norm_str(a))
    if t == "list":
        return (t, tuple(_cell(x) for x in a))            # list có thứ tự
    if t == "table":
        cols = [norm_str(c) for c in a["columns"]]
        order = sorted(range(len(cols)), key=lambda i: cols[i])   # cột đổi chỗ vẫn khớp
        rows = sorted(tuple(_cell(r[i]) for i in order) for r in a["rows"])  # hàng là tập
        return (t, tuple(cols[i] for i in order), tuple(rows))
    return (t, json.dumps(a, sort_keys=True, ensure_ascii=False))


def vote(cands):
    """cands: danh sách đáp án (dict {"type","answer"}), cands[0] là candidate greedy.
    Trả về (đáp_án_thắng, độ_đồng_thuận, chỉ_số_candidate_đại_diện).
    Candidate abstain (SQL lỗi, kết quả rỗng) không là phiếu cho đáp án nào nhưng vẫn
    nằm trong mẫu số: 2 phiếu trên 4 candidate -> độ đồng thuận 0.5.
    Hòa phiếu thì ưu tiên nhóm xuất hiện sớm hơn (tức nhóm có candidate greedy)."""
    n = len(cands)
    groups = {}  # khóa -> [số phiếu, chỉ số đầu tiên]
    for i, c in enumerate(cands):
        if c.get("type") == "abstain":
            continue
        g = groups.setdefault(answer_key(c), [0, i])
        g[0] += 1
    if not groups:
        return {"type": "abstain"}, 0.0, None
    votes, idx = max(groups.values(), key=lambda g: (g[0], -g[1]))
    return cands[idx], votes / n, idx


def apply_threshold(answer: dict, confidence: float, threshold: float) -> dict:
    """Chỉ trả lời khi độ đồng thuận >= ngưỡng; còn lại abstain (0 điểm thay vì -1)."""
    return answer if confidence >= threshold else {"type": "abstain"}