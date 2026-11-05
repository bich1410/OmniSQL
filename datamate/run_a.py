"""Nấc A: OmniSQL-7B sinh một SQL (greedy) -> chạy trên SQLite -> định dạng đáp án.
Không có abstain thông minh: SQL lỗi / kết quả rỗng thì abstain (điểm 0, không bị -1).

Dùng:
  python run_a.py --split_dir /kaggle/input/<ten-dataset>/public_test \
                  --out /kaggle/working/submission.jsonl --backend vllm
  (train: thêm --limit 300 để chạy thử một phần; rồi chấm bằng score.py)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from common import (build_db, build_prompt, extract_sql, read_jsonl, read_text, run_sql,
                    schema_text, to_answer, write_jsonl)

CHARS_PER_TOKEN = 2.2          # ước lượng thô cho tiếng Việt, an toàn hơn tiếng Anh
MAX_PROMPT_TOKENS = 6500       # model huấn luyện với ngữ cảnh 8192, chừa chỗ cho phần suy luận


def est_tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN)


def fit_prompt(tables_info, question, as_of, readme, rulebook):
    """Co dần các phần phụ cho tới khi prompt vừa ngân sách token."""
    for n_ex, rb_lim, rd_lim in [(2, 3000, 1500), (2, 1500, 800), (1, 800, 400), (0, 400, 0)]:
        schema = schema_text(tables_info["db"], tables_info["tables"], n_examples=n_ex)
        p = build_prompt(schema, question, as_of, readme[:rd_lim], rulebook[:rb_lim])
        if est_tokens(p) <= MAX_PROMPT_TOKENS:
            return p
    return p  # đành chịu; log sẽ cho biết


def generate(prompts, backend, model, batch_size, oracle=None, ids=None, tp=1, quant="none"):
    if backend == "stub":  # kiểm thử luồng, không cần GPU: đọc SQL có sẵn
        return [f"```sql\n{oracle.get(i, 'SELECT * FROM khong_co_bang')}\n```" for i in ids]
    if backend == "vllm":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        tok = AutoTokenizer.from_pretrained(model)
        llm = LLM(model=model, dtype="float16", tensor_parallel_size=tp, max_model_len=8192,
                  gpu_memory_utilization=0.90, swap_space=8, enforce_eager=True,
                  disable_custom_all_reduce=True, trust_remote_code=True)
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False) for p in prompts]
        outs = llm.generate(chats, SamplingParams(temperature=0, max_tokens=2048, n=1))
        return [o.outputs[0].text for o in outs]
    # backend "hf": chậm hơn nhưng ít lỗi cài đặt hơn (dùng khi vLLM không cài được)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    kw = {}
    if quant == "4bit":   # Colab T4 1 GPU: 7B fp16 (~15GB) không vừa, nén 4-bit còn ~5GB
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto", **kw)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        chats = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}],
                                         add_generation_prompt=True, tokenize=False) for i in idx]
        enc = tok(chats, return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=1024, do_sample=False)
        for j, i in enumerate(idx):
            res[i] = tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  hf: {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work_dir", default="/tmp/datamate_work")
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--model", default="seeklhy/OmniSQL-7B")
    ap.add_argument("--backend", choices=["vllm", "hf", "stub"], default="vllm")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1, help="số GPU chia cho vLLM (Kaggle T4x2 -> 2)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="chỉ backend hf")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (thử nhanh)")
    ap.add_argument("--oracle", default=None, help="(stub) json id -> SQL")
    a = ap.parse_args()

    split = Path(a.split_dir)
    work = Path(a.work_dir)
    log_dir = Path(a.log_dir) if a.log_dir else Path(a.out).parent / "logs"
    questions = read_jsonl(split / "questions.jsonl")
    if a.limit:
        questions = questions[: a.limit]
    rulebook = read_text(split / "rulebook.md", 20000)   # rulebook của CHÍNH split này (đề yêu cầu)

    t_start = time.time()
    by_biz = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)

    prompts, metas = [], []
    for biz, qs in by_biz.items():
        t0 = time.time()
        biz_dir = split / "data" / biz
        try:
            db, tables = build_db(biz_dir, work)
        except Exception as e:  # dữ liệu hỏng: abstain cả business, không làm sập cả lượt chạy
            print(f"[{biz}] LỖI nạp dữ liệu: {e}")
            for q in qs:
                metas.append({"q": q, "db": None, "err": f"build_db: {e}"})
                prompts.append("")
            continue
        info = {"db": db, "tables": tables}
        readme = read_text(biz_dir / "README.md", 20000)
        for q in qs:
            prompts.append(fit_prompt(info, q["question"], q.get("as_of_date", ""), readme, rulebook))
            metas.append({"q": q, "db": db, "err": None})
        print(f"[{biz}] {len(tables)} bảng, {len(qs)} câu, nạp {time.time() - t0:.1f}s")

    live = [i for i, m in enumerate(metas) if m["err"] is None]
    oracle = json.load(open(a.oracle, encoding="utf-8")) if a.oracle else None
    t_gen = time.time()
    outs = generate([prompts[i] for i in live], a.backend, a.model, a.batch_size,
                    oracle=oracle, ids=[metas[i]["q"]["id"] for i in live], tp=a.tp, quant=a.quant)
    gen_s = time.time() - t_gen
    raw = {i: o for i, o in zip(live, outs)}
    per_q = gen_s / max(1, len(live))

    submission, logs = [], []
    for i, m in enumerate(metas):
        q = m["q"]
        t0 = time.time()
        sql, ans, err = None, {"type": "abstain"}, m["err"]
        if err is None:
            sql = extract_sql(raw[i])
            if sql is None:
                err = "không tách được SQL"
            else:
                try:
                    cols, rows = run_sql(m["db"], sql)
                    ans = to_answer(q["question"], cols, rows)
                except Exception as e:
                    err = f"SQL lỗi: {e}"
        submission.append({"id": q["id"], **ans})
        logs.append({"id": q["id"], "answer": ans, "elapsed_s": round(per_q + time.time() - t0, 3),
                     "sql": sql, "error": err, "model_output": raw.get(i)})

    write_jsonl(Path(a.out), submission)
    write_jsonl(log_dir / "logs.jsonl", logs)
    n_abs = sum(1 for s in submission if s["type"] == "abstain")
    print(f"Xong {len(submission)} câu | abstain {n_abs} | {time.time() - t_start:.0f}s | -> {a.out}")


if __name__ == "__main__":
    main()
