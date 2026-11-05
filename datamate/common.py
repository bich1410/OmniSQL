"""Nấc A - phần dùng chung: nạp dữ liệu vào SQLite, dựng schema, dựng prompt OmniSQL,
chạy SQL và chuyển kết quả sang đáp án đúng định dạng của DataMate.

Dữ liệu của đề được mount CHỈ ĐỌC (Rule 5), nên mọi file tạm đều ghi vào work_dir.
"""
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- prompt OmniSQL
# Nguyên văn template trong README của repo RUCKBReasoning/OmniSQL.
PROMPT_TEMPLATE = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''


# ---------------------------------------------------------------- tiện ích chuỗi
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def snake(name: str) -> str:
    s = strip_accents(str(name)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = "c_" + s
    return s


def unique(names):
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}_{k}")
        seen[n] = k + 1
    return out


# ---------------------------------------------------------------- nạp dữ liệu
def smart_types(df: pd.DataFrame) -> pd.DataFrame:
    """Đọc CSV dạng chữ rồi chỉ đổi cột sang số khi an toàn: không mất số 0 đầu (mã, SĐT)."""
    for c in df.columns:
        col = df[c]
        nn = col.dropna().astype(str)
        if nn.empty or nn.str.match(r"^0\d").any():
            continue
        conv = pd.to_numeric(col, errors="coerce")
        if conv.notna().sum() == col.notna().sum():
            df[c] = conv
    return df


def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return smart_types(pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=True, low_memory=False))
        except UnicodeDecodeError:
            continue
    return smart_types(pd.read_csv(path, encoding="latin-1", dtype=str, low_memory=False))


def build_db(biz_dir: Path, work_dir: Path):
    """Trả về (đường dẫn db, danh sách bảng).
    Mỗi bảng: {"name", "cols": [(tên_sql, tên_gốc, kiểu)], "src"}"""
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / f"{biz_dir.name}.db"
    if db_path.exists():
        db_path.unlink()

    sqlite_files = sorted(p for p in biz_dir.iterdir() if p.suffix.lower() in (".sqlite", ".db", ".sqlite3"))
    if sqlite_files:  # lấy file sqlite đầu tiên làm nền (bản sao, vì thư mục gốc chỉ đọc)
        shutil.copy(sqlite_files[0], db_path)
    con = sqlite3.connect(db_path)

    used = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    frames = []  # (tên bảng, df, nguồn)
    for p in sorted(biz_dir.iterdir()):
        ext = p.suffix.lower()
        if ext == ".csv":
            frames.append((p.stem, read_csv_any(p), p.name))
        elif ext in (".xlsx", ".xlsm", ".xls"):
            sheets = {k: smart_types(v) for k, v in pd.read_excel(p, sheet_name=None, dtype=str).items()}
            for sh, df in sheets.items():
                nm = p.stem if len(sheets) == 1 else f"{p.stem}_{sh}"
                frames.append((nm, df, f"{p.name}[{sh}]"))

    origmap = {}  # tên bảng -> {tên cột sql: tên cột gốc trong file}
    for nm, df, src in frames:
        tname = snake(nm)
        while tname in used:
            tname += "_x"
        used.add(tname)
        orig = [str(c) for c in df.columns]
        new = unique([snake(c) for c in orig])
        origmap[tname] = dict(zip(new, orig))
        df.columns = new
        df.to_sql(tname, con, index=False)

    tables = []
    for (tname,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        om = origmap.get(tname, {})
        cols = [(r[1], om.get(r[1], r[1]), (r[2] or "text").lower())
                for r in con.execute(f'PRAGMA table_info("{tname}")')]
        tables.append({"name": tname, "cols": cols})
    con.commit()
    con.close()
    return db_path, tables


def _short(v, n=40):
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def schema_text(db_path: Path, tables, n_examples=2) -> str:
    """Schema dạng CREATE TABLE + chú thích 'example: [...]' (đúng kiểu OmniSQL được huấn luyện)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for t in tables:
        tn = t["name"]
        decls = []
        for cname, orig_name, ctype in t["cols"]:
            ty = "integer" if "int" in ctype else "real" if ctype in ("real", "float", "double", "numeric") else "text"
            try:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{cname}" FROM "{tn}" WHERE "{cname}" IS NOT NULL LIMIT {n_examples}')]
            except sqlite3.Error:
                vals = []
            ex = ", ".join(repr(_short(v)) if isinstance(v, str) else str(v) for v in vals)
            notes = []
            if orig_name != cname:
                notes.append(f"original header: {orig_name}")
            if ex:
                notes.append(f"example: [{ex}]")
            decls.append((f"{cname} {ty}", ", ".join(notes)))
        body = []
        for i, (d, c) in enumerate(decls):
            comma = "," if i < len(decls) - 1 else ""
            body.append(f"    {d}{comma}" + (f" -- {c}" if c else ""))
        out.append(f"CREATE TABLE {tn} (\n" + "\n".join(body) + "\n);")
    con.close()
    return "\n".join(out)


def read_text(path: Path, limit: int) -> str:
    try:
        s = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return s if len(s) <= limit else s[:limit] + "\n...[cắt bớt]"


def build_prompt(schema: str, question: str, as_of: str, readme: str, rulebook: str) -> str:
    """Ghi chú, rulebook và as_of_date đi vào phần 'external knowledge' của câu hỏi (README OmniSQL cho phép)."""
    knowledge = (
        f"Today's date (as-of date) is {as_of}. Use it to resolve relative dates such as "
        f"'tháng này', 'tuần rồi', 'hôm qua'.\n"
        f"Business notes (README):\n{readme}\n"
        f"Business rules (rulebook):\n{rulebook}"
    )
    q = f"{knowledge}\n\nQuestion: {question}"
    return PROMPT_TEMPLATE.format(db_details=schema, question=q)


# ---------------------------------------------------------------- tách & chạy SQL
def extract_sql(text: str):
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    if not blocks:
        return None
    sql = blocks[-1].strip()
    return sql or None


def run_sql(db_path: Path, sql: str, timeout_s=30, max_rows=2000):
    """Chạy SQL chỉ-đọc. Trả về (cột, hàng) hoặc ném Exception."""
    head = sql.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with") or head.startswith("--")):
        raise ValueError("chỉ cho phép SELECT/WITH")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    import time
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > timeout_s else 0, 100000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
    finally:
        con.close()
    return cols, rows


# ---------------------------------------------------------------- kết quả -> đáp án
def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _fmt_number(x: float):
    return int(x) if float(x).is_integer() else round(x, 2)


def _iso_date(s: str):
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        return s
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})[ T].*", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def to_answer(question: str, cols, rows):
    """Heuristic nấc A. Trả về dict {"type", "answer"} hoặc {"type": "abstain"}."""
    qn = strip_accents(question).lower()
    if not rows:
        return {"type": "abstain"}
    if len(cols) == 1:
        vals = [r[0] for r in rows]
        if len(vals) == 1:
            v = vals[0]
            if v is None:
                return {"type": "abstain"}
            n = _num(v)
            if n is not None:
                if re.search(r"%|phan tram|ti le|ty le", qn):
                    return {"type": "percent", "answer": round(n, 2)}
                if n in (0.0, 1.0) and re.search(r"\bkhong\s*\??\s*$", qn):
                    return {"type": "boolean", "answer": bool(n)}
                return {"type": "number", "answer": _fmt_number(n)}
            if isinstance(v, str):
                d = _iso_date(v)
                if d and re.search(r"ngay|thang|luc nao|khi nao", qn):
                    return {"type": "date", "answer": d}
                return {"type": "string", "answer": v}
            return {"type": "abstain"}
        if any(v is None for v in vals) or len(vals) > 100:
            return {"type": "abstain"}
        items = [(_fmt_number(_num(v)) if _num(v) is not None else v) for v in vals]
        return {"type": "list", "answer": items}
    # nhiều cột -> bảng nhỏ
    if len(rows) > 100:
        return {"type": "abstain"}
    out_rows = [[(_fmt_number(_num(c)) if _num(c) is not None else c) for c in r] for r in rows]
    return {"type": "table", "answer": {"columns": list(cols), "rows": out_rows}}


# ---------------------------------------------------------------- I/O jsonl
def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
