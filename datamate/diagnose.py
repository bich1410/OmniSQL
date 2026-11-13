"""Chẩn đoán từng câu: đáp án chuẩn so với dự đoán (greedy), SQL, lỗi -> <run_dir>/diagnose.txt.

  python diagnose.py --run_dir runs/base_n1_train100 --ref answers.jsonl --questions questions.jsonl

Mục đích: tìm LỖI HỆ THỐNG (sai thang số, sai định dạng, sai bảng...) mà điểm tổng hợp không cho thấy.
Với câu kiểu number/percent sai, in thêm tỉ lệ dự đoán / chuẩn để phát hiện sai thang (x1000, x100...).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from score import is_unans, load, score_one


def short(x, n=70):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def ratio(pred, ref):
    try:
        p, r = float(pred["answer"]), float(ref["answer"])
        return f"x{p / r:.3g}" if r else "ref=0"
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return "-"


def write(run_dir, ref_path, q_path):
    run = Path(run_dir)
    ref, qs = load(ref_path), load(q_path)
    logs = {}
    for line in open(run / "logs" / "logs.jsonl", encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            logs[d["id"]] = d
    rows = []
    for qid, lg in logs.items():
        r = ref.get(qid)
        if r is None:
            continue
        c0 = lg["candidates"][0] if lg["candidates"] else {"sql": None, "answer": {"type": "abstain"}, "error": None}
        pred = c0["answer"]
        s, lab = score_one(pred, r)
        rt = "unanswerable" if is_unans(r) else r["type"]
        why = ""
        if lab != "correct":
            if c0["error"]:
                why = "SQL_ERR"
            elif pred.get("type") == "abstain":
                why = "EMPTY"
            elif pred.get("type") != rt:
                why = f"TYPE({pred.get('type')}!={rt})"
            elif rt in ("number", "percent"):
                why = "VALUE " + ratio(pred, r)
            else:
                why = "VALUE"
        rows.append((lab, rt, why, qid, qs.get(qid, {}).get("question", ""), r, pred, c0))
    order = {"wrong": 0, "abstain": 1, "correct": 2}
    rows.sort(key=lambda x: (order[x[0]], x[1], x[3]))
    cnt = Counter((x[0], x[2].split(" ")[0] or "-") for x in rows)
    out = ["TỔNG HỢP (nhãn, nguyên nhân) -> số câu:"]
    out += [f"  {k}: {v}" for k, v in sorted(cnt.items())]
    ratios = Counter(x[2] for x in rows if x[2].startswith("VALUE x"))
    if ratios:
        out.append("Tỉ lệ dự đoán/chuẩn ở câu number/percent sai (phổ biến nhất): "
                   + ", ".join(f"{k[6:]}:{v}" for k, v in ratios.most_common(8)))
    out.append("")
    for lab, rt, why, qid, q, r, pred, c0 in rows:
        out.append(f"[{lab.upper()}] {qid} {rt} {why}")
        out.append(f"  Q: {short(q, 130)}")
        out.append(f"  chuẩn: {short(r.get('answer'))} | dự đoán: {short(pred.get('answer', pred.get('type')))}")
        out.append(f"  SQL: {short(c0.get('sql') or '', 260)}")
        if c0.get("error"):
            out.append(f"  lỗi: {short(c0['error'], 120)}")
    (run / "diagnose.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    return cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--questions", required=True)
    a = ap.parse_args()
    write(a.run_dir, a.ref, a.questions)
