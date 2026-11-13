"""Bước 2: bung chữ viết tắt.

2a. Tên cột viết tắt (SL, ĐVT, HĐ, PTTT...) -> thêm chú thích nghĩa vào schema ("meaning: ...").
2b. Câu hỏi kiểu chat (sp, dc, ko...) -> chuẩn hóa chữ viết tắt trước khi đưa vào prompt.

LƯU Ý: từ điển dưới đây là BẢN KHỞI ĐẦU, viết từ kiến thức chung về sổ sách bán lẻ Việt Nam
và vài ví dụ trong đề (SL, ĐVT, HĐ, PTTT); chưa đối chiếu với toàn bộ dữ liệu thật. Chữ nào chưa
có trong từ điển sẽ được liệt kê bởi `list_uncovered()` (run_all.py in ra đầu mỗi lần chạy)
để bổ sung. Chữ nhập nhằng ghi nhiều nghĩa, cách nhau bằng " / " để model tự chọn theo giá trị mẫu.
Chỉ áp dụng cho TÊN CỘT và CÂU HỎI, tuyệt đối không đụng vào giá trị trong bảng.
"""
import re
import unicodedata
from collections import Counter

# khóa: viết tắt đã bỏ dấu, chữ thường
COLUMN_ABBREV = {
    "sl": "số lượng", "dvt": "đơn vị tính", "hd": "hóa đơn", "pttt": "phương thức thanh toán",
    "sp": "sản phẩm", "kh": "khách hàng", "nv": "nhân viên", "dg": "đơn giá",
    "ck": "chiết khấu / chuyển khoản", "tt": "thành tiền / thanh toán",
    "nsx": "ngày sản xuất / nhà sản xuất", "hsd": "hạn sử dụng", "ncc": "nhà cung cấp",
    "sdt": "số điện thoại", "dc": "địa chỉ", "mh": "mặt hàng", "gv": "giá vốn", "gb": "giá bán",
    "dh": "đơn hàng", "stt": "số thứ tự", "mst": "mã số thuế", "dt": "doanh thu / điện thoại",
    "ln": "lợi nhuận", "tl": "tỷ lệ", "kl": "khối lượng", "cp": "chi phí", "vc": "vận chuyển",
    "pvc": "phí vận chuyển", "dvvc": "đơn vị vận chuyển", "cod": "thu hộ khi giao hàng (COD)",
    "dm": "danh mục", "lh": "lô hàng / liên hệ", "bh": "bảo hành / bán hàng",
    # bổ sung từ danh sách "chưa phủ" của dữ liệu train thật (dv 6 lần, hh 4, gd 3, cccd 3)
    "dv": "đơn vị / dịch vụ", "hh": "hàng hóa", "gd": "giao dịch", "cccd": "căn cước công dân",
    "cmnd": "chứng minh nhân dân", "tk": "tài khoản", "ds": "doanh số / danh sách",
}

# từ trong CÂU HỎI (giữ nguyên dấu nếu có), thay bằng dạng đầy đủ
QUESTION_ABBREV = {
    "sp": "sản phẩm", "dc": "được", "đc": "được", "ko": "không", "hok": "không", "k0": "không",
    "j": "gì", "trc": "trước", "vs": "với", "bn": "bao nhiêu", "ntn": "như thế nào",
    "sl": "số lượng", "kh": "khách hàng", "nv": "nhân viên", "dt": "doanh thu",
    "ln": "lợi nhuận", "hd": "hóa đơn", "ncc": "nhà cung cấp", "mh": "mặt hàng",
    "thg": "tháng", "hh": "hàng hóa", "gd": "giao dịch", "dv": "đơn vị",
}

# từ ngắn hay gặp nhưng KHÔNG phải viết tắt (để báo cáo "chưa phủ" đỡ nhiễu)
_COMMON_SHORT = {"ma", "ten", "gia", "so", "tien", "loai", "nhom", "ngay", "gio", "thang", "nam",
                 "id", "no", "co", "la", "va", "cua", "tu", "den", "ghi", "chu", "mo", "ta", "key",
                 "tax", "tong", "min", "max", "avg", "sum", "tel", "web", "sdt",
                 # từ tiếng Việt đầy đủ (bỏ dấu), độ dài <= 4, hay gặp trong tên cột
                 "hang", "kho", "don", "san", "pham", "khac", "mau", "size", "dia", "chi", "gio", "phut",
                 "giay", "tinh", "quan", "xa", "nha", "cong", "ty", "ban", "mua", "nhap", "xuat", "ton",
                 "thu", "tra", "cap", "tuoi", "cuoi", "dau", "moi", "cu", "huy", "dat", "giao", "nhan"}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def header_tokens(header: str):
    """'SL_ton' / 'Đơn giá (ĐVT)' / 'soLuong' -> các token chữ thường, bỏ dấu."""
    h = "".join((" " + c) if (i and c.isupper() and str(header)[i - 1].islower()) else c
                for i, c in enumerate(str(header)))              # camelCase: chữ thường rồi chữ hoa
    return [t for t in re.split(r"[^0-9a-z]+", strip_accents(h).lower()) if t]


def expand_header(header: str):
    """Chú thích cho một tên cột, ví dụ 'SL=số lượng; ĐVT=đơn vị tính'; None nếu không bung được gì."""
    out, seen = [], set()
    for t in header_tokens(header):
        if t in COLUMN_ABBREV and t not in seen:
            seen.add(t)
            out.append(f"{t.upper()}={COLUMN_ABBREV[t]}")
    return "; ".join(out) if out else None


def normalize_question(q: str) -> str:
    """Thay từ viết tắt nguyên từ trong câu hỏi; giữ nguyên mọi thứ khác (số, mã, tên, dấu câu)."""
    def sub(m):
        w = m.group(0)
        return QUESTION_ABBREV.get(w.lower(), w)
    return re.sub(r"\w+", sub, q)


def list_uncovered(headers, top=40, max_len=4):
    """Token ngắn (<= max_len ký tự chữ) trong tên cột mà từ điển chưa có -> [(token, số lần)]."""
    cnt = Counter()
    for h in headers:
        for t in header_tokens(h):
            if t.isalpha() and len(t) <= max_len and t not in COLUMN_ABBREV and t not in _COMMON_SHORT:
                cnt[t] += 1
    return cnt.most_common(top)
