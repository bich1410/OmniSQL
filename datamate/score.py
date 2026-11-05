"""Chấm điểm theo đúng bảng luật trong đề DataMate (mục Evaluation).

  python score.py --pred submission.jsonl --ref answers.jsonl [--questions questions.jsonl]

Mỗi câu: đúng (+1) / sai (-1) / abstain (0). Câu thiếu trong file nộp = abstain.
PA = trung bình điểm các câu. Câu hỏi unanswerable mà trả lời bất kỳ loại nào khác = sai.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict


def norm_str(s) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if (c.isalpha() or c.isdigit() or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def num_eq(a, b) -> bool:
    try:
        return round(float(a), 2) == round(float(b), 2)
    except (TypeError, ValueError):
        return False


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def elem_eq(p, r, aliases=()) -> bool:
    if is_num(r):
        return is_num(p) and num_eq(p, r)
    if is_num(p):
        return False
    accepted = {norm_str(r)} | {norm_str(a) for a in aliases}
    return norm_str(p) in accepted


def match(pred: dict, ref: dict) -> bool:
    t = ref["type"]
    if pred.get("type") != t:
        return False
    p, r = pred.get("answer"), ref.get("answer")
    alias = ref.get("alias") or []
    if t in ("number", "percent"):
        return is_num(p) and num_eq(p, r)
    if t == "string":
        return isinstance(p, str) and norm_str(p) in ({norm_str(r)} | {norm_str(a) for a in alias})
    if t == "date":
        return p == r
    if t == "boolean":
        return isinstance(p, bool) and p == r
    if t == "list":
        return isinstance(p, list) and len(p) == len(r) and all(elem_eq(x, y) for x, y in zip(p, r))
    if t == "table":
        try:
            pc, rc = p["columns"], r["columns"]
            prow, rrow = p["rows"], r["rows"]
        except (TypeError, KeyError):
            return False
        if len(pc) != len(rc):
            return False
        pn, rn = [norm_str(c) for c in pc], [norm_str(c) for c in rc]
        order = [pn.index(c) for c in rn] if sorted(pn) == sorted(rn) and len(set(pn)) == len(pn) else list(range(len(rc)))
        prow = [[row[i] for i in order] for row in prow]
        if len(prow) != len(rrow):
            return False

        def key(row):
            return tuple(("n", round(float(c), 2)) if is_num(c) else ("s", norm_str(c)) for c in row)
        return Counter(map(key, prow)) == Counter(map(key, rrow))
    if t == "unanswerable":
        return True
    return False


def is_unans(ref: dict) -> bool:
    return ref.get("type") == "unanswerable" or ref.get("answer") is None


def score_one(pred: dict | None, ref: dict):
    """Trả về (điểm, nhãn)."""
    if pred is None or pred.get("type") == "abstain":
        return 0, "abstain"
    if is_unans(ref):
        return (1, "correct") if pred.get("type") == "unanswerable" else (-1, "wrong")
    if pred.get("type") == "unanswerable":
        return -1, "wrong"
    return (1, "correct") if match(pred, ref) else (-1, "wrong")


def load(path):
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}


def evaluate(pred_path, ref_path, q_path=None, only_run=False):
    pred, ref = load(pred_path), load(ref_path)
    if only_run:  # chạy thử một phần (--limit): chỉ chấm các câu có mặt trong file nộp
        ref = {k: v for k, v in ref.items() if k in pred}
    qs = load(q_path) if q_path else {}
    rows = []
    for qid, r in ref.items():
        rt = "unanswerable" if is_unans(r) else r["type"]
        s, lab = score_one(pred.get(qid), r)
        rows.append({"id": qid, "ref_type": rt, "score": s, "label": lab,
                     "biz": qs.get(qid, {}).get("business_id", "?")})
    return rows


def summarize(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    out = {}
    for k, v in sorted(g.items()):
        c = sum(x["label"] == "correct" for x in v)
        w = sum(x["label"] == "wrong" for x in v)
        a = sum(x["label"] == "abstain" for x in v)
        out[k] = {"n": len(v), "correct": c, "wrong": w, "abstain": a,
                  "PA": round(sum(x["score"] for x in v) / len(v), 4),
                  "acc_when_answered": round(c / (c + w), 4) if c + w else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--only_run", action="store_true", help="chỉ chấm các id có trong file dự đoán")
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()
    rows = evaluate(a.pred, a.ref, a.questions, a.only_run)
    n = len(rows)
    c = sum(r["label"] == "correct" for r in rows)
    w = sum(r["label"] == "wrong" for r in rows)
    ab = sum(r["label"] == "abstain" for r in rows)
    pa = sum(r["score"] for r in rows) / n
    print(f"N={n} | đúng={c} sai={w} abstain={ab} | PA={pa:.4f}")
    print(f"độ phủ (tỉ lệ dám trả lời) = {(c + w) / n:.3f} | độ chính xác khi trả lời = {c / (c + w) if c + w else float('nan'):.3f}")
    by_type = summarize(rows, "ref_type")
    print("\nTheo kiểu đáp án chuẩn:")
    for k, v in by_type.items():
        print(f"  {k:13s} n={v['n']:4d} đúng={v['correct']:4d} sai={v['wrong']:4d} abstain={v['abstain']:4d} PA={v['PA']:+.3f}")
    by_biz = summarize(rows, "biz") if a.questions else {}
    if by_biz:
        print("\nTheo business (PA):")
        for k, v in by_biz.items():
            print(f"  {k:10s} n={v['n']:4d} PA={v['PA']:+.3f}")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"N": n, "correct": c, "wrong": w, "abstain": ab, "PA": pa,
                       "by_type": by_type, "by_business": by_biz}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
