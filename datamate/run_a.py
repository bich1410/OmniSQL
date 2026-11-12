"""Nấc A + Bước 1: OmniSQL-7B sinh n SQL (1 greedy + n-1 lấy mẫu) -> chạy trên SQLite
-> định dạng đáp án -> BỎ PHIẾU giữa các candidate -> abstain nếu độ đồng thuận thấp.
(--n 1 và không bật cờ nào = đúng hành vi nấc A cũ: một SQL greedy, SQL lỗi / kết quả rỗng thì abstain.)
Bước 2: --expand  bung chữ viết tắt (tên cột trong schema + câu hỏi kiểu chat)
Bước 3: --link    chọn bảng bằng embedding thay vì khớp từ khóa (--embedder bge|hash)

Dùng:
  python run_a.py --split_dir /kaggle/input/<ten-dataset>/public_test \
                  --out /kaggle/working/submission.jsonl --backend vllm --n 4
  (train: thêm --limit 300 để chạy thử một phần; chấm bằng score.py, chọn ngưỡng bằng calibrate.py)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from abbrev import normalize_question
from common import (build_db, build_prompt, extract_sql, read_jsonl, read_text, run_sql,
                    schema_text, select_rulebook, select_tables, to_answer, write_jsonl)
from vote import apply_threshold, vote

CHARS_PER_TOKEN = 2.2          # ước lượng thô cho tiếng Việt, an toàn hơn tiếng Anh
MAX_PROMPT_TOKENS = 6500       # model huấn luyện với ngữ cảnh 8192, chừa chỗ cho phần suy luận


def est_tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN)


def fit_prompt(tables_info, question, as_of, readme, rulebook, expand=False, scorer=None):
    """Co dần các phần phụ cho tới khi prompt vừa ngân sách token.
    Bảng chọn theo liên quan câu hỏi (select_tables) vì nhiều business có 20-28 bảng
    (mỗi file/sheet Excel một bảng) -> schema một mình đã vượt ngân sách.
    Rulebook chọn theo mục liên quan câu hỏi (select_rulebook), không cắt thô."""
    for max_t, n_ex, rb_lim, rd_lim in [(12, 2, 6000, 1500), (8, 1, 3000, 800),
                                         (6, 1, 1500, 400), (4, 0, 600, 0)]:
        tables = select_tables(question, tables_info["tables"], max_tables=max_t, scorer=scorer)
        schema = schema_text(tables_info["db"], tables, n_examples=n_ex, expand=expand)
        rb = select_rulebook(question, rulebook, rb_lim)
        p = build_prompt(schema, question, as_of, readme[:rd_lim], rb)
        if est_tokens(p) <= MAX_PROMPT_TOKENS:
            return p
    return p  # đành chịu; log sẽ cho biết


_ENGINES = {}   # nạp model một lần cho cả tiến trình (run_all.py chạy nhiều cấu hình liên tiếp)


def _vllm_engine(model, tp):
    if ("vllm", model, tp) not in _ENGINES:
        from transformers import AutoTokenizer
        from vllm import LLM
        tok = AutoTokenizer.from_pretrained(model)
        llm = LLM(model=model, dtype="float16", tensor_parallel_size=tp, max_model_len=8192,
                  gpu_memory_utilization=0.90, enforce_eager=True,
                  disable_custom_all_reduce=True, trust_remote_code=True)
        _ENGINES[("vllm", model, tp)] = (tok, llm)
    return _ENGINES[("vllm", model, tp)]


def _hf_engine(model, quant):
    if ("hf", model, quant) not in _ENGINES:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        tok.padding_side = "left"
        kw = {}
        if quant == "4bit":   # Colab T4 1 GPU: 7B fp16 (~15GB) không vừa, nén 4-bit còn ~5GB
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto", **kw)
        _ENGINES[("hf", model, quant)] = (tok, mdl)
    return _ENGINES[("hf", model, quant)]


def generate(prompts, backend, model, batch_size, n=4, temperature=0.7,
             oracle=None, ids=None, tp=1, quant="none"):
    """Mỗi prompt trả về danh sách n chuỗi: [0] = greedy (như nấc A), [1:] = n-1 bản lấy mẫu.
    k = n-1 bản lấy mẫu dùng temperature > 0 để các bản khác nhau, nếu không sẽ giống hệt greedy."""
    k = max(0, n - 1)
    if backend == "stub":  # kiểm thử luồng, không cần GPU: oracle[id] là một SQL hoặc danh sách SQL
        res = []
        for i in ids:
            sqls = oracle.get(i, "SELECT * FROM khong_co_bang")
            sqls = [sqls] if isinstance(sqls, str) else list(sqls)
            sqls = (sqls + [sqls[-1]] * n)[:n]      # thiếu thì lặp lại bản cuối
            res.append([f"```sql\n{s}\n```" for s in sqls])
        return res
    if backend == "vllm":
        from vllm import SamplingParams
        tok, llm = _vllm_engine(model, tp)
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False) for p in prompts]
        res = [[o.outputs[0].text] for o in
               llm.generate(chats, SamplingParams(temperature=0, max_tokens=2048, n=1))]
        if k:  # n=k: vLLM sinh k bản cho mỗi prompt, prefill (đọc prompt dài) chỉ tính một lần
            samp = llm.generate(chats, SamplingParams(temperature=temperature, top_p=0.95,
                                                      max_tokens=2048, n=k))
            for r, o in zip(res, samp):
                r.extend(c.text for c in o.outputs)
        return res
    # backend "hf": chậm hơn nhưng ít lỗi cài đặt hơn (dùng khi vLLM không cài được)
    import torch
    tok, mdl = _hf_engine(model, quant)
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
            res[i] = [tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True)]
        if k:  # num_return_sequences=k: mỗi đầu vào sinh k bản, xếp liền nhau trong out
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=1024, do_sample=True, temperature=temperature,
                                   top_p=0.95, num_return_sequences=k)
            for j, i in enumerate(idx):
                res[i].extend(tok.decode(out[j * k + t][enc.input_ids.shape[1]:], skip_special_tokens=True)
                              for t in range(k))
        print(f"  hf: {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    return res


def spread_sample(questions, limit):
    """Lấy `limit` câu xen kẽ đều giữa các business (thay vì N câu đầu có thể chỉ thuộc 1-2 business)."""
    by = defaultdict(list)
    for q in questions:
        by[q["business_id"]].append(q)
    pools = [list(by[b]) for b in sorted(by)]
    out = []
    while len(out) < limit and any(pools):
        for p in pools:
            if p and len(out) < limit:
                out.append(p.pop(0))
    return out


def load_or_build(biz_dir, work, reuse):
    """build_db có cache: nạp dữ liệu vào SQLite tốn thời gian, các lần chạy ablation dùng lại được."""
    meta = work / f"{biz_dir.name}.tables.json"
    if reuse and meta.exists() and (work / f"{biz_dir.name}.db").exists():
        d = json.load(open(meta, encoding="utf-8"))
        return Path(d["db"]), [{"name": t["name"], "cols": [tuple(c) for c in t["cols"]]} for t in d["tables"]]
    db, tables = build_db(biz_dir, work)
    json.dump({"db": str(db), "tables": tables}, open(meta, "w", encoding="utf-8"), ensure_ascii=False)
    return db, tables


def build_parser():
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
    ap.add_argument("--oracle", default=None, help="(stub) json id -> SQL hoặc danh sách SQL")
    ap.add_argument("--n", type=int, default=4, help="số candidate mỗi câu (1 greedy + n-1 lấy mẫu); 1 = nấc A cũ")
    ap.add_argument("--temperature", type=float, default=0.7, help="nhiệt độ cho các bản lấy mẫu")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="chỉ trả lời khi độ đồng thuận >= ngưỡng. 0.5 chỉ là giá trị TẠM, chọn bằng calibrate.py")
    ap.add_argument("--expand", action="store_true", help="Bước 2: bung viết tắt (tên cột + câu hỏi)")
    ap.add_argument("--link", action="store_true", help="Bước 3: chọn bảng bằng embedding")
    ap.add_argument("--embedder", choices=["bge", "hash"], default="bge", help="model embedding cho --link")
    ap.add_argument("--spread", action="store_true", help="--limit lấy câu xen kẽ đều giữa các business")
    ap.add_argument("--reuse_db", action="store_true", help="dùng lại SQLite đã nạp ở work_dir (cho nhiều lần chạy)")
    return ap


def run(a):

    split = Path(a.split_dir)
    work = Path(a.work_dir)
    log_dir = Path(a.log_dir) if a.log_dir else Path(a.out).parent / "logs"
    questions = read_jsonl(split / "questions.jsonl")
    if a.limit:
        questions = spread_sample(questions, a.limit) if a.spread else questions[: a.limit]
    rulebook = read_text(split / "rulebook.md", 20000)   # rulebook của CHÍNH split này (đề yêu cầu)

    t_start = time.time()
    by_biz = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)

    linker = None
    if a.link:
        from link import TableLinker, get_embedder
        linker = TableLinker(get_embedder(a.embedder), expand=a.expand)

    prompts, metas = [], []
    for biz, qs in by_biz.items():
        t0 = time.time()
        biz_dir = split / "data" / biz
        try:
            db, tables = load_or_build(biz_dir, work, a.reuse_db)
        except Exception as e:  # dữ liệu hỏng: abstain cả business, không làm sập cả lượt chạy
            print(f"[{biz}] LỖI nạp dữ liệu: {e}")
            for q in qs:
                metas.append({"q": q, "db": None, "err": f"build_db: {e}"})
                prompts.append("")
            continue
        info = {"db": db, "tables": tables}
        readme = read_text(biz_dir / "README.md", 20000)
        for q in qs:
            qtext = normalize_question(q["question"]) if a.expand else q["question"]   # chỉ cho prompt;
            scorer = linker.scorer(biz, tables, qtext) if linker else None             # to_answer vẫn dùng câu gốc
            prompts.append(fit_prompt(info, qtext, q.get("as_of_date", ""), readme, rulebook,
                                      expand=a.expand, scorer=scorer))
            metas.append({"q": q, "db": db, "err": None})
        print(f"[{biz}] {len(tables)} bảng, {len(qs)} câu, nạp {time.time() - t0:.1f}s")

    live = [i for i, m in enumerate(metas) if m["err"] is None]
    oracle = json.load(open(a.oracle, encoding="utf-8")) if a.oracle else None
    t_gen = time.time()
    outs = generate([prompts[i] for i in live], a.backend, a.model, a.batch_size, n=a.n,
                    temperature=a.temperature, oracle=oracle,
                    ids=[metas[i]["q"]["id"] for i in live], tp=a.tp, quant=a.quant)
    gen_s = time.time() - t_gen
    raw = {i: o for i, o in zip(live, outs)}     # raw[i] = danh sách n chuỗi, [0] là greedy
    per_q = gen_s / max(1, len(live))

    submission, logs, cache = [], [], {}
    for i, m in enumerate(metas):
        q = m["q"]
        t0 = time.time()
        cands, cand_logs = [], []                 # đáp án + nhật ký từng candidate
        for text in raw.get(i, []):
            sql, ans, err = extract_sql(text), {"type": "abstain"}, None
            if sql is None:
                err = "không tách được SQL"
            else:
                key = (str(m["db"]), sql)         # nhiều candidate trùng SQL thì chỉ chạy một lần
                if key not in cache:              # chỉ cache kết quả chạy; to_answer còn phụ thuộc câu hỏi
                    try:
                        cache[key] = run_sql(m["db"], sql)
                    except Exception as e:
                        cache[key] = e
                res_or_err = cache[key]
                if isinstance(res_or_err, Exception):
                    err = f"SQL lỗi: {res_or_err}"
                else:
                    ans = to_answer(q["question"], *res_or_err)
            cands.append(ans)
            cand_logs.append({"sql": sql, "answer": ans, "error": err})
        if m["err"] is not None or not cands:    # lỗi nạp dữ liệu -> abstain cả câu
            voted, conf, win = {"type": "abstain"}, 0.0, None
        else:
            voted, conf, win = vote(cands)
        ans = apply_threshold(voted, conf, a.threshold)
        submission.append({"id": q["id"], **ans})
        # "sql"/"error"/"model_output" giữ như nấc A (candidate greedy); "candidates" để calibrate.py quét ngưỡng
        logs.append({"id": q["id"], "answer": ans, "voted_answer": voted, "confidence": round(conf, 4),
                     "winner": win, "elapsed_s": round(per_q + time.time() - t0, 3),
                     "sql": cand_logs[0]["sql"] if cand_logs else None,
                     "error": m["err"] or (cand_logs[0]["error"] if cand_logs else None),
                     "model_output": raw[i][0] if i in raw else None, "candidates": cand_logs})

    write_jsonl(Path(a.out), submission)
    write_jsonl(log_dir / "logs.jsonl", logs)
    n_abs = sum(1 for s in submission if s["type"] == "abstain")
    json.dump({**vars(a), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t_start)},
              open(log_dir / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # ghi lại cấu hình lần chạy
    print(f"Xong {len(submission)} câu | abstain {n_abs} | {time.time() - t_start:.0f}s | -> {a.out}")


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()