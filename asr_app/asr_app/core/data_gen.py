"""Sinh bộ dữ liệu MÔ PHỎNG giãn nở ASR để chạy thử phần mềm (Chương 5.1).

CẢNH BÁO: đây KHÔNG phải dữ liệu thí nghiệm thật. Hàm sinh dữ liệu dùng một
công thức bán thực nghiệm đơn giản (đường cong S theo thời gian) chỉ nhằm
minh họa quy trình. Khi có dữ liệu thật (CSV theo đúng từ điển biến), hãy đặt
vào data/real_asr.csv, ứng dụng sẽ ưu tiên dùng tệp này.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "fly_ash_pct",       # % tro bay thay thế theo khối lượng chất kết dính
    "alkali_kg_m3",      # kiềm quy đổi Na2O tương đương, kg/m3
    "reactive_agg_pct",  # % cốt liệu phản ứng trong tổng cốt liệu
    "w_b",               # tỷ lệ nước / chất kết dính
    "binder_kg_m3",      # tổng chất kết dính, kg/m3
    "temp_c",            # nhiệt độ, °C
    "rh_pct",            # độ ẩm tương đối, %
    "age_days",          # tuổi mẫu, ngày
]
TARGET = "expansion_pct"

RANGES = {
    "fly_ash_pct": (0.0, 50.0),
    "alkali_kg_m3": (2.5, 6.0),
    "reactive_agg_pct": (0.0, 100.0),
    "w_b": (0.35, 0.60),
    "binder_kg_m3": (300.0, 500.0),
    "temp_c": (20.0, 60.0),
    "rh_pct": (70.0, 100.0),
    "age_days": (0.0, 730.0),
}


def _expansion(fa, alk, reac, wb, binder, temp, rh, t):
    # Tro bay giảm giãn nở nhưng có hiệu quả giảm dần và phụ thuộc kiềm (phi tuyến)
    fa_eff = 1.0 - 0.9 * (1 - np.exp(-fa / 18.0)) * (1.0 / (1.0 + 0.12 * np.clip(alk - 3.0, 0, None)))
    max_exp = 0.45 * (reac / 100.0) ** 0.8 * (alk / 4.0) ** 1.2 * fa_eff
    max_exp *= (1.0 + 0.6 * (wb - 0.45)) * np.clip((rh - 70.0) / 30.0, 0, 1)
    rate = 0.012 * np.exp(0.045 * (temp - 20.0))      # nhiệt độ đẩy nhanh phản ứng
    t_half = 1.0 / rate * 2.2
    return max_exp / (1.0 + np.exp(-(t - t_half) * rate * 1.8))


def generate(n_mixes: int = 220, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.array([0, 7, 14, 28, 56, 90, 120, 180, 270, 365, 540, 730], dtype=float)
    rows = []
    for i in range(n_mixes):
        p = {k: rng.uniform(*RANGES[k]) for k in RANGES if k != "age_days"}
        p["fly_ash_pct"] = rng.choice([0, 10, 15, 20, 25, 30, 40, 50]) + rng.normal(0, 1.0)
        p["fly_ash_pct"] = float(np.clip(p["fly_ash_pct"], 0, 50))
        for t in times:
            e = _expansion(p["fly_ash_pct"], p["alkali_kg_m3"], p["reactive_agg_pct"],
                           p["w_b"], p["binder_kg_m3"], p["temp_c"], p["rh_pct"], t)
            e = max(0.0, e + rng.normal(0, 0.004))
            rows.append({"mix_id": f"M{i:03d}", **p, "age_days": t, TARGET: e})
    df = pd.DataFrame(rows)
    df["source"] = "SIMULATED"
    return df


if __name__ == "__main__":
    generate().to_csv("data/simulated_asr.csv", index=False)
    print("Đã ghi data/simulated_asr.csv")
