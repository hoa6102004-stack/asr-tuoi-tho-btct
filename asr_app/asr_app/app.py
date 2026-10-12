"""
Dự đoán tuổi thọ kết cấu công trình BTCT trên cơ sở độ bền
chống ăn mòn kiềm cốt liệu của bê tông sử dụng tro bay.

Chạy ứng dụng:
    streamlit run app.py
"""

import io
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_gen import FEATURES, RANGES
from core.lifespan import time_to_threshold
from core.model import (
    load_artifacts,
    out_of_domain,
    predict_curve,
    train_all,
)


# =========================================================
# 1. CẤU HÌNH GIAO DIỆN
# =========================================================

APP_TITLE = (
    "Dự đoán tuổi thọ kết cấu công trình BTCT trên cơ sở "
    "độ bền chống ăn mòn kiềm cốt liệu của bê tông sử dụng tro bay"
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧱",
    layout="wide",
)

LABELS = {
    "fly_ash_pct": "Tro bay thay thế (% chất kết dính)",
    "alkali_kg_m3": "Kiềm quy đổi Na₂O tương đương (kg/m³)",
    "reactive_agg_pct": "Cốt liệu phản ứng (%)",
    "w_b": "Tỷ lệ nước/chất kết dính",
    "binder_kg_m3": "Tổng chất kết dính (kg/m³)",
    "temp_c": "Nhiệt độ (°C)",
    "rh_pct": "Độ ẩm tương đối (%)",
}

DB = "history.db"


# =========================================================
# 2. QUẢN LÝ LỊCH SỬ DỰ ĐOÁN
# =========================================================

def db_init():
    """Khởi tạo cơ sở dữ liệu lịch sử dự đoán."""
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                ts TEXT,
                model TEXT,
                version TEXT,
                inputs TEXT,
                eps_lim REAL,
                eps_at_age REAL,
                status TEXT
            )
        """)


def db_add(row):
    """Lưu một kết quả dự đoán."""
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?)",
            row,
        )


def db_read():
    """Đọc lịch sử dự đoán."""
    with sqlite3.connect(DB) as conn:
        return pd.read_sql(
            "SELECT * FROM history ORDER BY ts DESC",
            conn,
        )


# =========================================================
# 3. NẠP MÔ HÌNH HỌC MÁY
# =========================================================

@st.cache_resource
def artifacts():
    """Nạp mô hình đã huấn luyện; huấn luyện nếu chưa có."""
    if not os.path.exists("models/models.joblib"):
        train_all()

    return load_artifacts()


db_init()
models, meta = artifacts()


# =========================================================
# 4. THANH ĐIỀU HƯỚNG
# =========================================================

page = st.sidebar.radio(
    "Màn hình chức năng",
    [
        "1 · Tổng quan",
        "2 · Nhập thông số",
        "3 · Kết quả dự đoán",
        "4 · Báo cáo",
    ],
)

st.sidebar.caption(
    f"Phiên bản mô hình: {meta['version']}"
)

if meta["data_source"] != "REAL":
    st.sidebar.warning(
        "Đang sử dụng DỮ LIỆU MÔ PHỎNG. "
        "Kết quả chỉ có giá trị minh họa."
    )


# =========================================================
# 5. MÀN HÌNH TỔNG QUAN
# =========================================================

if page.startswith("1"):
    st.title("🧱 Dự đoán tuổi thọ kết cấu công trình BTCT trên cơ sở độ bền chống ăn mòn kiềm cốt liệu của bê tông sử dụng tro bay")

    st.markdown("""
    ### Giới thiệu đề tài

    Phần mềm ứng dụng học máy để phân tích khả năng chống
    phản ứng kiềm – silic (ASR) của bê tông sử dụng tro bay,
    từ đó hỗ trợ đánh giá nguy cơ suy giảm độ bền của kết cấu
    bê tông cốt thép (BTCT).
    """)

    if meta["data_source"] != "REAL":
        st.error(
            "Dữ liệu hiện tại là dữ liệu mô phỏng, không phải "
            "dữ liệu thí nghiệm thực tế. Cần thay bằng dữ liệu "
            "được kiểm chứng trước khi sử dụng cho nghiên cứu."
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mô hình tốt nhất",
        meta["best_model"],
    )

    c2.metric(
        "Số cấp phối",
        str(meta["n_mixes"]),
    )

    c3.metric(
        "Nguồn dữ liệu",
        meta["data_source"],
    )

    st.subheader("Mục tiêu và phạm vi")

    st.markdown("""
    **Các chức năng chính của phần mềm:**

    1. Nhập thông số cấp phối bê tông sử dụng tro bay.
    2. Phân tích ảnh hưởng của vật liệu và điều kiện môi trường.
    3. Dự đoán độ giãn nở ASR theo thời gian bằng mô hình học máy.
    4. Ước tính thời điểm đạt ngưỡng giãn nở được lựa chọn.
    5. So sánh kết quả giữa các mô hình học máy.
    6. Lưu lịch sử và xuất báo cáo kết quả.

    **Giới hạn khoa học:**

    Kết quả dự đoán độ giãn nở ASR hoặc thời điểm đạt ngưỡng
    không tự động tương đương với tuổi thọ thực tế của công trình.
    Muốn xác định tuổi thọ kết cấu BTCT cần có mô hình liên hệ
    đã được hiệu chuẩn và kiểm chứng bằng dữ liệu thí nghiệm,
    điều kiện hiện trường và các cơ chế xuống cấp liên quan.
    """)

    st.subheader("So sánh các mô hình học máy")

    if os.path.exists("models/comparison.csv"):
        comp = pd.read_csv("models/comparison.csv")
        st.dataframe(
            comp.round(4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Chưa tìm thấy models/comparison.csv. "
            "Hãy kiểm tra lại quá trình huấn luyện mô hình."
        )

    st.caption(
        "Nên lựa chọn mô hình dựa trên kết quả kiểm định "
        "và đánh giá độc lập, không chỉ dựa trên dữ liệu huấn luyện."
    )

    with st.expander("Huấn luyện lại mô hình"):

        if st.button("Huấn luyện lại toàn bộ mô hình"):
            with st.spinner("Đang huấn luyện mô hình học máy..."):
                train_all()
                st.cache_resource.clear()

            st.success(
                "Đã hoàn thành huấn luyện. "
                "Hãy tải lại trang để nạp mô hình mới."
            )

        if "domain" in meta:
            st.write(
                "Phạm vi dữ liệu mô hình:",
                pd.DataFrame(
                    meta["domain"],
                    index=["Giá trị nhỏ nhất", "Giá trị lớn nhất"],
                ).T,
            )


# =========================================================
# 6. MÀN HÌNH NHẬP THÔNG SỐ
# =========================================================

elif page.startswith("2"):

    st.title("Nhập thông số vật liệu và môi trường")

    mode = st.radio(
        "Phương thức nhập dữ liệu",
        [
            "Nhập thủ công",
            "Tải CSV nhiều cấp phối",
        ],
        horizontal=True,
    )

    if mode == "Nhập thủ công":

        with st.form("input_parameters"):

            st.subheader("Thông số cấp phối bê tông")

            cols = st.columns(2)
            vals = {}

            for i, feature in enumerate(LABELS):

                lo, hi = meta["default_ranges"][feature]

                old_value = (
                    st.session_state.get("params", {}).get(feature)
                )

                if old_value is None:
                    old_value = (lo + hi) / 2

                vals[feature] = cols[i % 2].number_input(
                    LABELS[feature],
                    min_value=0.0,
                    max_value=float(hi),
                    value=float(
                        min(max(old_value, lo), hi)
                    ),
                    step=float(max((hi - lo) / 100, 0.001)),
                    key=f"input_{feature}",
                )

            st.subheader("Thông số đánh giá ASR")

            c1, c2, c3 = st.columns(3)

            eps_lim = c1.number_input(
                "Ngưỡng giãn nở ε_lim (%)",
                min_value=0.001,
                max_value=5.0,
                value=float(
                    st.session_state.get("eps_lim", 0.10)
                ),
                step=0.01,
                help=(
                    "Lựa chọn ngưỡng phù hợp với tiêu chuẩn "
                    "và phương pháp thử áp dụng."
                ),
            )

            cur_age = c2.number_input(
                "Tuổi hiện tại (ngày)",
                min_value=0.0,
                max_value=730.0,
                value=float(
                    st.session_state.get("cur_age", 0.0)
                ),
                step=10.0,
            )

            horizon = c3.number_input(
                "Dự đoán đến (ngày)",
                min_value=30.0,
                max_value=730.0,
                value=float(
                    st.session_state.get("horizon", 730.0)
                ),
                step=30.0,
            )

            submitted = st.form_submit_button(
                "Lưu thông số",
                type="primary",
            )

            if submitted:

                errors = []

                if vals["w_b"] <= 0:
                    errors.append(
                        "Tỷ lệ nước/chất kết dính phải lớn hơn 0."
                    )

                if (
                    vals["fly_ash_pct"] > 100
                    or vals["reactive_agg_pct"] > 100
                    or vals["rh_pct"] > 100
                ):
                    errors.append(
                        "Các giá trị phần trăm không được vượt 100."
                    )

                if errors:
                    for error in errors:
                        st.error(error)

                else:
                    st.session_state["params"] = vals
                    st.session_state["eps_lim"] = eps_lim
                    st.session_state["cur_age"] = cur_age
                    st.session_state["horizon"] = horizon

                    st.success(
                        "Đã lưu thông số. "
                        "Chuyển sang màn hình 3 để dự đoán."
                    )

        if "params" in st.session_state:

            warnings = out_of_domain(
                st.session_state["params"],
                meta,
            )

            for warning in warnings:
                st.warning("⚠ " + warning)

    else:

        st.write(
            "Các cột dữ liệu bắt buộc:",
            ", ".join(
                feature
                for feature in FEATURES
                if feature != "age_days"
            ),
        )

        uploaded_file = st.file_uploader(
            "Chọn tệp dữ liệu CSV",
            type=["csv"],
        )

        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

            required = [
                feature
                for feature in FEATURES
                if feature != "age_days"
            ]

            missing = [
                feature
                for feature in required
                if feature not in df.columns
            ]

            if missing:
                st.error(
                    f"Tệp CSV thiếu các cột: {missing}"
                )

            elif df[required].isna().any().any():
                st.error(
                    "Dữ liệu có giá trị trống. "
                    "Hãy kiểm tra và bổ sung trước khi dự đoán."
                )

            else:

                model_name = st.selectbox(
                    "Chọn mô hình học máy",
                    list(models),
                    index=list(models).index(
                        meta["best_model"]
                    ),
                )

                if st.button("Dự đoán dữ liệu CSV"):

                    rows = []

                    for _, row in df.iterrows():

                        parameters = {
                            feature: float(row[feature])
                            for feature in required
                        }

                        time_points = np.arange(
                            0,
                            731,
                            30,
                        )

                        prediction = predict_curve(
                            models[model_name],
                            parameters,
                            time_points,
                        )

                        rows.append({
                            **parameters,
                            "eps_730d_pct": float(
                                prediction[-1]
                            ),
                        })

                    result_df = pd.DataFrame(rows)

                    st.dataframe(
                        result_df.round(4),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Tải kết quả CSV",
                        result_df.to_csv(index=False).encode(
                            "utf-8-sig"
                        ),
                        "ket_qua_du_doan.csv",
                        "text/csv",
                    )


# =========================================================
# 7. MÀN HÌNH KẾT QUẢ DỰ ĐOÁN
# =========================================================

elif page.startswith("3"):

    st.title("Kết quả dự đoán và đánh giá ASR")

    if "params" not in st.session_state:
        st.info(
            "Vui lòng nhập và lưu thông số tại màn hình 2 trước."
        )
        st.stop()

    params = st.session_state["params"]
    eps_lim = st.session_state["eps_lim"]
    cur_age = st.session_state["cur_age"]
    horizon = st.session_state["horizon"]

    selected_models = st.multiselect(
        "Chọn mô hình để so sánh",
        list(models),
        default=[meta["best_model"]],
    )

    if not selected_models:
        st.warning("Hãy chọn ít nhất một mô hình.")
        st.stop()

    warnings = out_of_domain(params, meta)

    for warning in warnings:
        st.warning(
            "⚠ Dữ liệu ngoài phạm vi huấn luyện: " + warning
        )

    # Tạo đường cong dự đoán
    time_points = np.arange(
        0,
        horizon + 1,
        5.0,
    )

    fig = go.Figure()
    curves = {}

    for model_name in selected_models:

        curve = predict_curve(
            models[model_name],
            params,
            time_points,
        )

        curves[model_name] = curve

        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=curve,
                name=model_name,
                mode="lines",
            )
        )

    fig.add_hline(
        y=eps_lim,
        line_dash="dash",
        annotation_text=f"Ngưỡng ε_lim = {eps_lim:.3f}%",
    )

    fig.update_layout(
        title="Đường cong dự đoán độ giãn nở ASR",
        xaxis_title="Tuổi bê tông (ngày)",
        yaxis_title="Độ giãn nở ASR (%)",
        height=480,
        legend_title="Mô hình học máy",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # So sánh độ phân tán dự đoán giữa các mô hình
    all_curves = np.array([
        predict_curve(model, params, time_points)
        for model in models.values()
    ])

    spread = all_curves.max(axis=0) - all_curves.min(axis=0)

    st.caption(
        f"Độ phân tán giữa các mô hình tại "
        f"{horizon:.0f} ngày: {spread[-1]:.4f}%. "
        "Đây chỉ là dấu hiệu khác biệt giữa các mô hình, "
        "không phải khoảng tin cậy thống kê."
    )

    # Tính thời điểm đạt ngưỡng
    rows = []

    for model_name in selected_models:

        curve = curves[model_name]

        result = time_to_threshold(
            time_points,
            curve,
            eps_lim,
            cur_age,
        )

        evaluation_age = (
            cur_age if cur_age > 0 else horizon
        )

        eps_at_age = float(
            np.interp(
                evaluation_age,
                time_points,
                curve,
            )
        )

        rows.append({
            "Mô hình": model_name,
            "Độ giãn nở tại mốc (%)": round(
                eps_at_age, 4
            ),
            "Thời điểm đạt ngưỡng (ngày)": (
                None
                if result.t_star is None
                else round(result.t_star)
            ),
            "Trạng thái": result.status,
            "Diễn giải": result.message,
        })

        db_add((
            datetime.now().isoformat(timespec="seconds"),
            model_name,
            meta["version"],
            str(params),
            eps_lim,
            eps_at_age,
            result.status,
        ))

    result_df = pd.DataFrame(rows)

    st.subheader("Bảng kết quả")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )

    st.session_state["last_result"] = result_df

    st.info("""
    **Lưu ý khi diễn giải kết quả**

    Thời điểm đạt ngưỡng ASR là kết quả của mô hình dự đoán
    theo các giả thiết và dữ liệu đầu vào. Không được xem
    trực tiếp giá trị này là tuổi thọ thực tế của kết cấu BTCT.

    Muốn dự đoán tuổi thọ công trình cần xây dựng và kiểm chứng
    mô hình liên hệ giữa ASR, mức độ hư hỏng và tiêu chí tuổi thọ
    kết cấu bằng dữ liệu phù hợp.
    """)


# =========================================================
# 8. MÀN HÌNH BÁO CÁO
# =========================================================

else:

    st.title("Báo cáo dự đoán")

    result_df = st.session_state.get("last_result")

    if result_df is None:

        st.info(
            "Chưa có kết quả. "
            "Hãy thực hiện dự đoán tại màn hình 3."
        )

    else:

        params = st.session_state["params"]

        lines = [
            f"# {APP_TITLE}",
            "",
            f"Ngày lập báo cáo: {datetime.now():%d/%m/%Y %H:%M}",
            f"Phiên bản mô hình: {meta['version']}",
            f"Nguồn dữ liệu: {meta['data_source']}",
            "",
            "## 1. Thông số đầu vào",
        ]

        for key, value in params.items():
            lines.append(
                f"- {LABELS[key]}: {value}"
            )

        lines.extend([
            "",
            "## 2. Tiêu chí dự đoán",
            f"- Ngưỡng giãn nở: {st.session_state['eps_lim']}%",
            f"- Tuổi hiện tại: {st.session_state['cur_age']} ngày",
            f"- Thời hạn dự đoán: {st.session_state['horizon']} ngày",
            "",
            "## 3. Kết quả",
            result_df.to_string(index=False),
            "",
            "## 4. Giới hạn áp dụng",
            "- Phân tích tập trung vào phản ứng kiềm – silic (ASR).",
            "- Chưa đại diện cho mọi cơ chế xuống cấp của kết cấu BTCT.",
            "- Thời điểm đạt ngưỡng ASR không đồng nghĩa với tuổi thọ công trình.",
            "- Dự đoán ngoài phạm vi huấn luyện có thể không đáng tin cậy.",
        ])

        if meta["data_source"] != "REAL":
            lines.append(
                "- Dữ liệu huấn luyện là dữ liệu mô phỏng; "
                "kết quả chỉ dùng minh họa."
            )

        report = "\n".join(lines)

        st.markdown(report)

        st.download_button(
            "Tải báo cáo Markdown",
            data=report,
            file_name="bao_cao_du_doan_tuoi_tho_btct.md",
            mime="text/markdown",
        )

        csv_buffer = io.StringIO()
        result_df.to_csv(
            csv_buffer,
            index=False,
        )

        st.download_button(
            "Tải kết quả CSV",
            data=csv_buffer.getvalue(),
            file_name="ket_qua_du_doan_asr.csv",
            mime="text/csv",
        )

    st.subheader("Lịch sử dự đoán")

    st.dataframe(
        db_read(),
        use_container_width=True,
        hide_index=True,
    )