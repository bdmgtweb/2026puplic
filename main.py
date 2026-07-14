from pathlib import Path

code = r'''from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="대한민국 쌍둥이 지역 찾기",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(88,101,242,0.10), transparent 25%),
                radial-gradient(circle at 90% 20%, rgba(0,184,148,0.10), transparent 25%);
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 900;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #5b61f6, #00a884);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sub-title {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .info-card {
            border: 1px solid rgba(128,128,128,0.20);
            border-radius: 18px;
            padding: 18px;
            background: rgba(255,255,255,0.04);
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
        }

        .small-text {
            color: #6b7280;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 16px;
            padding: 12px;
            background: rgba(255,255,255,0.03);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "-": np.nan}),
        errors="coerce",
    ).fillna(0)


def find_csv_file():
    base_dir = Path(__file__).resolve().parent
    csv_files = sorted(base_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "main.py와 같은 폴더에서 CSV 파일을 찾을 수 없습니다."
        )

    preferred_files = [
        file
        for file in csv_files
        if "연령별인구현황" in file.name or "인구현황" in file.name
    ]

    return preferred_files[0] if preferred_files else csv_files[0]


def read_csv_safely(file_path):
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"CSV 파일을 읽지 못했습니다. 마지막 오류: {last_error}"
    )


def extract_region_info(region_text):
    text = str(region_text).strip()
    match = re.match(r"^(.*?)\s*\((\d+)\)\s*$", text)

    if match:
        region_name = re.sub(r"\s+", " ", match.group(1)).strip()
        region_code = match.group(2)
        return region_name, region_code

    return re.sub(r"\s+", " ", text).strip(), ""


def classify_region_level(code, region_name):
    code = str(code)

    if len(code) == 10 and code.isdigit():
        if code[2:] == "00000000":
            return "시도"

        if code[5:] == "00000":
            return "시군구"

        return "읍면동"

    word_count = len(str(region_name).split())

    if word_count <= 1:
        return "시도"

    if word_count == 2:
        return "시군구"

    return "읍면동"


def get_parent_regions(region_name):
    parts = str(region_name).split()
    sido = parts[0] if parts else ""
    sigungu = " ".join(parts[:2]) if len(parts) >= 2 else sido
    return sido, sigungu


def age_sort_key(label):
    number = re.search(r"\d+", str(label))
    return int(number.group()) if number else 999


def make_age_group(age, interval):
    if age >= 100:
        return "100세 이상"

    start = (age // interval) * interval
    end = min(start + interval - 1, 99)
    return f"{start}~{end}세"


def cosine_similarity_matrix(matrix, target):
    matrix_norm = np.linalg.norm(matrix, axis=1)
    target_norm = np.linalg.norm(target)

    denominator = matrix_norm * target_norm
    denominator = np.where(denominator == 0, 1, denominator)

    return np.dot(matrix, target) / denominator


def population_similarity(populations, target_population):
    populations = np.maximum(populations.astype(float), 1)
    target_population = max(float(target_population), 1)

    log_difference = np.abs(np.log(populations / target_population))
    return np.exp(-log_difference)


def format_population(value):
    return f"{int(round(float(value))):,}명"


def percentage_change(value, base):
    if float(base) == 0:
        return 0

    return (float(value) - float(base)) / float(base) * 100


def age_range_total(row, available_ages, start_age, end_age):
    columns = [
        f"전체_{age}세"
        for age in available_ages
        if start_age <= age <= end_age and f"전체_{age}세" in row.index
    ]

    if not columns:
        return 0

    return float(pd.to_numeric(row[columns], errors="coerce").fillna(0).sum())


def demographic_summary(row, available_ages):
    total = max(float(row["총인구수"]), 1)

    children = age_range_total(row, available_ages, 0, 14)
    working = age_range_total(row, available_ages, 15, 64)
    elderly = age_range_total(row, available_ages, 65, 100)

    return {
        "유소년": children / total * 100,
        "생산연령": working / total * 100,
        "고령": elderly / total * 100,
        "고령화지수": elderly / max(children, 1) * 100,
    }


@st.cache_data(show_spinner="인구 데이터를 불러오고 있습니다...")
def load_and_prepare_data():
    csv_path = find_csv_file()
    raw = read_csv_safely(csv_path)

    raw.columns = [str(column).strip() for column in raw.columns]

    if "행정구역" not in raw.columns:
        raise ValueError("CSV 파일에서 '행정구역' 열을 찾을 수 없습니다.")

    region_info = raw["행정구역"].apply(extract_region_info)

    raw["지역명"] = region_info.apply(lambda value: value[0])
    raw["행정구역코드"] = region_info.apply(lambda value: value[1])

    raw["행정단계"] = raw.apply(
        lambda row: classify_region_level(
            row["행정구역코드"],
            row["지역명"],
        ),
        axis=1,
    )

    parent_info = raw["지역명"].apply(get_parent_regions)
    raw["시도"] = parent_info.apply(lambda value: value[0])
    raw["시군구"] = parent_info.apply(lambda
