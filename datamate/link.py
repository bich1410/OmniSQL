"""Bước 3: nối câu hỏi với đúng BẢNG bằng embedding (thay cách khớp từ khóa trong select_tables).

Embedding = model nhỏ đổi một câu/cụm từ thành vector số; nghĩa gần nhau thì vector gần nhau.
Điểm của một bảng = độ giống (cosine) giữa câu hỏi và "mô tả bảng" (tên bảng + tiêu đề các cột
+ nghĩa đã bung ở Bước 2 nếu bật). Chạy trên CPU để không tranh bộ nhớ GPU với OmniSQL.

Phạm vi: CHỈ chọn bảng (chỉ có tác dụng khi business có nhiều hơn max_tables bảng). Chưa nối giá trị trong ô.
"""
import hashlib
import re

import numpy as np

from abbrev import expand_header, strip_accents


class HashEmbedder:
    """Embedding thô bằng băm n-gram ký tự (đã bỏ dấu). KHÔNG cần tải model: dùng để kiểm thử
    hoặc làm phương án dự phòng. Chỉ bắt được độ giống về mặt chữ, không bắt được nghĩa."""
    dim = 1024

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = " " + re.sub(r"\s+", " ", strip_accents(t).lower()) + " "
            for n in (3, 4):
                for j in range(len(s) - n + 1):
                    h = int(hashlib.md5(s[j:j + n].encode()).hexdigest()[:8], 16)
                    out[i, h % self.dim] += 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-9)


class BGEEmbedder:
    """BAAI/bge-m3 (đa ngôn ngữ, ~0,57B tham số). Vector = trạng thái ẩn của token [CLS], chuẩn hóa L2
    (đúng cách dùng embedding dense của BGE-M3). Dùng transformers có sẵn, không cần cài thêm gói."""

    def __init__(self, name="BAAI/bge-m3", device="cpu", max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device, self.max_length = device, max_length

    def encode(self, texts, batch_size=16):
        outs = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s:s + batch_size], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = self.model(**enc).last_hidden_state[:, 0]
            h = self.torch.nn.functional.normalize(h, dim=-1)
            outs.append(h.cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 1), dtype=np.float32)


def get_embedder(name: str):
    return HashEmbedder() if name == "hash" else BGEEmbedder()


def table_text(t, expand=False):
    """Mô tả một bảng để embedding: tên bảng + tiêu đề gốc các cột (+ nghĩa bung nếu bật Bước 2)."""
    parts = [t["name"].replace("_", " ")]
    for _sql_name, orig, _ty in t["cols"]:
        parts.append(str(orig))
        if expand:
            m = expand_header(str(orig))
            if m:
                parts.append(m)
    return " | ".join(parts)


class TableLinker:
    def __init__(self, embedder, expand=False):
        self.e, self.expand = embedder, expand
        self._cache = {}          # (khóa business, tên bảng) -> vector

    def scorer(self, biz_key, tables, question):
        """Trả về hàm t -> điểm (cosine), dùng làm tham số `scorer` của select_tables."""
        todo = [t for t in tables if (biz_key, t["name"]) not in self._cache]
        if todo:
            vecs = self.e.encode([table_text(t, self.expand) for t in todo])
            for t, v in zip(todo, vecs):
                self._cache[(biz_key, t["name"])] = v
        qv = self.e.encode([question])[0]
        return lambda t: float(np.dot(self._cache[(biz_key, t["name"])], qv))
