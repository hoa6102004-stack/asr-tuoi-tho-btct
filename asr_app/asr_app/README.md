# ASR – Dự đoán giãn nở phản ứng kiềm–silica của bê tông tro bay

Phần mềm Streamlit xây dựng theo Chương 5–10 của đề cương.

## Chạy trong VS Code
```bash
cd asr_app
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m core.model            # huấn luyện + so sánh mô hình
python -m pytest -q tests       # kiểm thử (Chương 10)
streamlit run app.py
```

## Cấu trúc
- `core/data_gen.py` – sinh dữ liệu **mô phỏng** (chỉ để chạy thử)
- `core/model.py` – huấn luyện Linear/RF/XGBoost/ANN, chia tập theo cấp phối, cảnh báo ngoài miền
- `core/lifespan.py` – tìm t* khi ε(t*)=ε_lim, xử lý "chưa đạt"/"đã vượt"
- `app.py` – 4 màn hình: Tổng quan, Nhập thông số, Kết quả, Báo cáo + lịch sử SQLite
- `tests/` – test case cho thuật toán ngưỡng và mô hình

## Dùng dữ liệu thật
Đặt `data/real_asr.csv` với các cột: `mix_id, fly_ash_pct, alkali_kg_m3, reactive_agg_pct, w_b,
binder_kg_m3, temp_c, rh_pct, age_days, expansion_pct`, rồi chạy lại `python -m core.model`.
Không trộn thanh vữa gia tốc với mẫu bê tông dài hạn nếu chưa hiệu chuẩn (Chương 3.6).

## Lưu ý
Bộ dữ liệu mặc định là mô phỏng; mọi chỉ số R²/MAE/RMSE chỉ phản ánh khả năng học
công thức mô phỏng, **không** có giá trị khoa học về bê tông thật.
