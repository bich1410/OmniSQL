"""Chọn ngưỡng đồng thuận bằng số liệu: quét ngưỡng trên log của TRAIN rồi chấm PA.

  python calibrate.py --logs /kaggle/working/logs/logs.jsonl \
                      --ref  <train>/answers.jsonl --questions <train>/questions.jsonl

In ra: (1) PA nếu chỉ dùng candidate greedy (= nấc A, mốc để so sánh),
       (2) PA khi bỏ phiếu ở từng ngưỡng, kèm độ phủ và độ chính xác khi trả lời.
Nếu PA bỏ phiếu không vượt mốc greedy thì bỏ phiếu chưa có ích ở cấu hình này.
Chạy với log chia theo business để thấy ngưỡng có ổn định giữa các cửa hàng không.
"""
import argparse
import json
from collections import defaultdict

from score import load, score_one
from vote import apply_threshold


def stats(rows):
    n = len(rows)
    c = sum(s == 1 for s in rows)
    w = sum(s == -1 for s in rows)
    return {"n": n, "PA": sum(rows) / n if n else float("nan"),
            "coverage": (c + w) / n if n else float("nan"),
            "acc": c / (c + w) if c + w else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--thresholds", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args()

    ref = load(a.ref)
    qs = load(a.questions) if a.questions else {}
    logs = [json.loads(l) for l in open(a.logs, encoding="utf-8") if l.strip()]
    logs = [l for l in logs if l["id"] in ref]            # chỉ chấm các câu có đáp án (và đã chạy)

    def pa_of(pred_fn, group=None):
        scores = defaultdict(list)
        for l in logs:
            s, _ = score_one(pred_fn(l), ref[l["id"]])
            scores["all"].append(s)
            if group:
                scores[qs.get(l["id"], {}).get("business_id", "?")].append(s)
        return scores

    base = pa_of(lambda l: l["candidates"][0]["answer"] if l["candidates"] else {"type": "abstain"})
    b = stats(base["all"])
    print(f"N={len(logs)} | mốc greedy (nấc A): PA={b['PA']:+.4f} phủ={b['coverage']:.3f} chính xác={b['acc']:.3f}")
    print(f"{'ngưỡng':>8} {'PA':>8} {'phủ':>7} {'chính xác':>10}  PA - mốc")
    best = None
    for th in [float(x) for x in a.thresholds.split(",")]:
        sc = pa_of(lambda l, th=th: apply_threshold(l["voted_answer"], l["confidence"], th), group=True)
        s = stats(sc["all"])
        print(f"{th:8.2f} {s['PA']:+8.4f} {s['coverage']:7.3f} {s['acc']:10.3f}  {s['PA'] - b['PA']:+.4f}")
        if best is None or s["PA"] > best[1]:
            best = (th, s["PA"], sc)
    print(f"\nNgưỡng tốt nhất trên log này: {best[0]} (PA={best[1]:+.4f})")
    if qs:
        print("PA theo business ở ngưỡng đó (xem có ổn định không):")
        for k, v in sorted(best[2].items()):
            if k != "all":
                print(f"  {k:10s} n={len(v):4d} PA={sum(v) / len(v):+.3f}")


if __name__ == "__main__":
    main()