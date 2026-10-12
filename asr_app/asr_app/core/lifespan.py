"""Suy ra thời gian đạt ngưỡng từ đường cong giãn nở dự đoán (Chương 4.5, 8.4).

Lưu ý khoa học: t* chỉ là "thời gian đạt ngưỡng THEO MÔ HÌNH và điều kiện giả định",
KHÔNG tự động là tuổi thọ công trình. Chỉ quy đổi sang năm khi người dùng xác nhận
đã có hệ số hiệu chuẩn hiện trường.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ThresholdResult:
    status: str                 # 'reached' | 'not_reached' | 'already_exceeded'
    t_star: Optional[float]     # ngày
    message: str


def smooth_monotone(eps):
    """Đường cong ASR tích lũy không giảm: lấy cummax để loại nhiễu mô hình."""
    return np.maximum.accumulate(np.asarray(eps, dtype=float))


def time_to_threshold(times, eps, eps_lim: float, current_age: float = 0.0) -> ThresholdResult:
    times = np.asarray(times, dtype=float)
    eps = smooth_monotone(eps)
    if eps_lim <= 0:
        raise ValueError("Ngưỡng giãn nở phải > 0.")

    eps_now = float(np.interp(current_age, times, eps))
    if eps_now >= eps_lim:
        return ThresholdResult(
            "already_exceeded", None,
            f"Tại tuổi {current_age:.0f} ngày, độ giãn nở dự đoán ({eps_now:.3f}%) đã "
            f"đạt/vượt ngưỡng {eps_lim:.3f}%. Cần xem xét/kiểm định.")

    if eps[-1] < eps_lim:
        return ThresholdResult(
            "not_reached", None,
            f"Chưa đạt ngưỡng trong khoảng dự đoán (đến {times[-1]:.0f} ngày). "
            "Điều này KHÔNG có nghĩa là tuổi thọ vô hạn.")

    i = int(np.argmax(eps >= eps_lim))          # chỉ số đầu tiên vượt ngưỡng
    t0, t1, e0, e1 = times[i - 1], times[i], eps[i - 1], eps[i]
    t_star = t1 if e1 == e0 else t0 + (eps_lim - e0) * (t1 - t0) / (e1 - e0)
    return ThresholdResult(
        "reached", float(t_star),
        f"Đường cong dự đoán đạt ngưỡng {eps_lim:.3f}% tại ≈ {t_star:.0f} ngày "
        f"(≈ {t_star/365.25:.1f} năm) trong điều kiện mô hình đã giả định.")


def to_service_years(t_star_days: float, field_factor: float) -> float:
    """Quy đổi sang năm hiện trường: CHỈ gọi khi có hệ số hiệu chuẩn hợp lệ."""
    return t_star_days * field_factor / 365.25
