import sys; sys.path.insert(0, "..")
from abbrev import expand_header, header_tokens, list_uncovered, normalize_question

# --- tách token tên cột
assert header_tokens("SL_ton") == ["sl", "ton"]
assert header_tokens("Đơn giá (ĐVT)") == ["don", "gia", "dvt"]
assert header_tokens("soLuong") == ["so", "luong"]

# --- 2a: bung tên cột
assert expand_header("SL") == "SL=số lượng"
assert expand_header("ĐVT") == "ĐVT=đơn vị tính".replace("Đ", "D")        # khóa chuẩn hóa bỏ dấu, in hoa không dấu
assert expand_header("Số HĐ") == "HD=hóa đơn"
assert expand_header("PTTT") == "PTTT=phương thức thanh toán"
assert expand_header("SL_DG") == "SL=số lượng; DG=đơn giá"
assert expand_header("ten_san_pham") is None                               # không viết tắt -> không chú thích
assert expand_header("SL SL") == "SL=số lượng"                              # không lặp
assert "chiết khấu / chuyển khoản" in expand_header("CK")                   # nhập nhằng: giữ cả hai nghĩa

# --- 2b: chuẩn hóa câu hỏi
assert normalize_question("sp bán dc nhiều nhất tuần rồi là gì") == "sản phẩm bán được nhiều nhất tuần rồi là gì"
assert normalize_question("Doanh thu hôm nay ko bằng hôm qua à?") == "Doanh thu hôm nay không bằng hôm qua à?"
assert normalize_question("SP001 bán bn cái") == "SP001 bán bao nhiêu cái"      # mã SP001 không bị đụng
assert normalize_question("giảm 10k thì sao") == "giảm 10k thì sao"            # 'k' không bị đổi
assert normalize_question("Câu hỏi bình thường, không có viết tắt.") == "Câu hỏi bình thường, không có viết tắt."

# --- báo cáo chữ chưa phủ: chỉ token ngắn chưa có trong từ điển
unc = dict(list_uncovered(["SL", "XYZ", "XYZ", "ma_hang", "ten", "abcdefgh"]))
assert unc == {"xyz": 2}, unc
print("test_abbrev đạt")
