import sys; sys.path.insert(0, "..")
import numpy as np
from common import select_tables
from link import HashEmbedder, TableLinker, table_text


def T(name, *cols):
    return {"name": name, "cols": [(c.lower(), c, "text") for c in cols]}


# 14 bảng, vượt max_tables=12: 2 bảng lớn (>6 cột) liên quan câu hỏi nằm giữa nhiều bảng lớn nhiễu
big_noise = [T(f"nhat_ky_{i}", *[f"cot{j}" for j in range(8)]) for i in range(8)]
target = T("phieu_nhap_kho", "Ngày nhập", "Mã SP", "SL nhập", "Giá nhập", "NCC", "Kho", "Ghi chú")
small = [T(f"danh_muc_{i}", "ma", "ten") for i in range(5)]
tables = big_noise + [target] + small                     # 14 bảng
assert len(tables) > 12

emb = HashEmbedder()
v = emb.encode(["abc", "abc", "xyz"])
assert np.allclose(np.linalg.norm(v, axis=1), 1.0) and np.allclose(v[0], v[1]) and not np.allclose(v[0], v[2])

linker = TableLinker(emb, expand=True)
q = "tháng này nhập kho bao nhiêu SL từ NCC"
scorer = linker.scorer("biz1", tables, q)
assert scorer(target) > max(scorer(t) for t in big_noise)           # bảng đúng điểm cao hơn bảng nhiễu

# select_tables: có scorer -> giữ bảng đúng; luôn giữ bảng nhỏ (danh mục); không vượt max_tables
picked = select_tables(q, tables, max_tables=10, scorer=scorer)
names = {t["name"] for t in picked}
assert "phieu_nhap_kho" in names and len(picked) == 10 and all(f"danh_muc_{i}" in names for i in range(5))
# không truyền scorer -> hành vi cũ (khớp từ khóa), vẫn trả đủ 10 bảng
assert len(select_tables(q, tables, max_tables=10)) == 10
# ít bảng hơn max_tables -> trả nguyên, không cần scorer
assert select_tables(q, tables[:5], max_tables=10, scorer=scorer) == tables[:5]

# cache: gọi lại không encode lại bảng
n_cached = len(linker._cache)
linker.scorer("biz1", tables, "câu khác")
assert len(linker._cache) == n_cached
# expand=True thêm nghĩa vào mô tả bảng
assert "SL=số lượng" in table_text(target, expand=True) and "SL=" not in table_text(target, expand=False)
print("test_link đạt")
