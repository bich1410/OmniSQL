"""Khảo sát dữ liệu thô: mỗi business, mỗi nhóm bảng, mỗi cột -> mẫu giá trị + dạng (9=chữ số, a=chữ).
Dùng để biết cần chuẩn hóa gì khi nạp (số có dấu phẩy, nhiều định dạng ngày...). Chỉ đọc, không cần GPU.
  python inspect_data.py <train_dir> <out.txt>"""
import collections
import re
import sys
from pathlib import Path

import pandas as pd


def shape(v):
    s = re.sub(r"\d", "9", str(v))
    s = re.sub(r"[^\W\d_]", "a", s)
    return re.sub(r"a+", "a", re.sub(r"9+", "9", s))[:24]


def frames(biz):
    for p in sorted(biz.iterdir()):
        e = p.suffix.lower()
        try:
            if e == ".csv":
                yield p.name, pd.read_csv(p, dtype=str, encoding="utf-8-sig", low_memory=False)
            elif e in (".xlsx", ".xlsm", ".xls"):
                for sh, df in pd.read_excel(p, sheet_name=None, dtype=str).items():
                    yield f"{p.name}[{sh}]", df
        except Exception as ex:
            yield p.name, ex


def main():
    train, out = Path(sys.argv[1]), sys.argv[2]
    lines = []
    for biz in sorted((train / "data").iterdir()):
        if not biz.is_dir():
            continue
        lines.append(f"\n===== {biz.name} =====")
        seen = {}
        for name, df in frames(biz):
            if isinstance(df, Exception):
                lines.append(f"{name}: LỖI {df}")
                continue
            stem = re.sub(r"\d+", "#", name)
            seen[stem] = seen.get(stem, 0) + 1
            if seen[stem] > 1:
                continue
            lines.append(f"-- {name} ({len(df)} dòng, {len(df.columns)} cột)")
            for c in df.columns:
                col = df[c].dropna().astype(str)
                sh = collections.Counter(shape(v) for v in col).most_common(3)
                lines.append(f"   {str(c)[:30]!r}: {list(col.head(3))!r:.70} | {sh}")
        lines.append("   (số bảng cùng mẫu tên: " + str({k: v for k, v in seen.items() if v > 1}) + ")")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
