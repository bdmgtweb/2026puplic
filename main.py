from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(
    page_title="대한민국 쌍둥이 지역 찾기",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS 디자인
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(88, 101, 242, 0.10), transparent 25%),
                radial-gradient(circle at 90% 20%, rgba(0, 184, 148, 0.10), transparent 25%);
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
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 18px;
            padding: 18px;
            background: rgba(255, 255, 255, 0.04);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        }

        .rank-card {
            border-radius: 16px;
            padding: 17px;
            margin-bottom: 10px;
            border: 1px solid rgba(128, 128, 128, 0.20);
            background: rgba(91, 97, 246, 0.05);
        }

        .rank-number {
            font-size: 1.7rem;
            font-weight: 900;
            color: #5b61f6;
        }

        .rank-name {
            font-size: 1.05rem;
            font-weight: 800;
        }

        .small-text {
            color: #6b7280;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 16px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.03);
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 공통 함수
# =========================================================
def clean_number(series: pd.Series) -> pd.Series:
    """쉼표가 포함된 문자열 숫자를 실수형으로 변환합니다."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "-": np.nan}),
        errors="coerce",
    ).fillna(0)


def find_csv_file() -> Path:
    """
    main.py와 같은 폴더에서 CSV 파일을 자동으로 찾습니다.
    '연령별인구현황'이 포함된 파일을 우선 선택합니다.
    """
    base_dir = Path(__file__).resolve().parent
    csv_files = list(base_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "main.py와 같은 폴더에서 CSV 파일을 찾을 수 없습니다."
        )

    preferred_files = [
        file
        for file in csv_files
        if "연령별인구현황" in file.name or "인구현황" in file.name
    ]

    if preferred_files:
        return preferred_files[0]

    return csv_files[0]


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """한국 공공데이터에서 자주 쓰이는 인코딩을 순서대로 시도합니다."""
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


def extract_region_info(region_text: str) -> tuple[str, str]:
    """행정구역 문자열에서 지역명과 행정구역 코드를 분리합니다."""
    text = str(region_text).strip()

    match = re.match(r"^(.*?)\s*\((\d+)\)\s*$", text)

    if match:
        region_name = re.sub(r"\s+", " ", match.group(1)).strip()
        region_code = match.group(2)
        return region_name, region_code

    return re.sub(r"\s+", " ", text).strip(), ""


def classify_region_level(code: str, region_name: str) -> str:
    """
    행정구역 코드 형태를 이용하여 시도·시군구·읍면동 단계를 구분합니다.
    """
    code = str(code)

    if len(code) != 10 or not code.isdigit():
        word_count = len(str(region_name).split())

        if word_count <= 1:
            return "시도"
        if word_count == 2:
            return "시군구"
        return "읍면동"

    if code[2:] == "00000000":
        return "시도"

    if code[5:] == "00000":
        return "시군구"

    return "읍면동"


def get_parent_regions(region_name: str) -> tuple[str, str]:
    """지역명에서 상위 시도와 시군구 명칭을 추출합니다."""
    parts = str(region_name).split()

    sido = parts[0] if parts else ""
    sigungu = " ".join(parts[:2]) if len(parts) >= 2 else sido

    return sido, sigungu


def age_sort_key(label: str) -> int:
    """연령구간 문자열 정렬용 함수입니다."""
    number = re.search(r"\d+", str(label))
    return int(number.group()) if number else 999


def make_age_group(age: int, interval: int) -> str:
    """나이를 지정된 간격의 연령구간으로 변환합니다."""
    if age >= 100:
        return "100세 이상"

    start = (age // interval) * interval
    end = min(start + interval - 1, 99)

    return f"{start}~{end}세"


def cosine_similarity_matrix(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """각 행과 선택 지역 벡터 사이의 코사인 유사도를 계산합니다."""
    matrix_norm = np.linalg.norm(matrix, axis=1)
    target_norm = np.linalg.norm(target)

    denominator = matrix_norm * target_norm
    denominator = np.where(denominator == 0, 1, denominator)

    return np.dot(matrix, target) / denominator


def population_similarity(populations: np.ndarray, target_population: float) -> np.ndarray:
    """
    인구 규모 유사도입니다.
    두 지역의 인구 비율 차이를 로그 척도로 비교합니다.
    """
    populations = np.maximum(populations.astype(float), 1)
    target_population = max(float(target_population), 1)

    log_difference = np.abs(
        np.log(populations / target_population)
    )

    return np.exp(-log_difference)


def format_population(value: float) -> str:
    return f"{int(round(value)):,}명"


def percentage_change(value: float, base: float) -> float:
    if base == 0:
        return 0
    return ((value - base) / base) * 100


# =========================================================
# 데이터 로딩 및 전처리
# =========================================================
@st.cache_data(show_spinner="인구 데이터를 불러오고 있습니다...")
def load_and_prepare_data() -> tuple[
    pd.DataFrame,
    list[str],
    dict[int, str],
    dict[int, str],
    Path,
]:
    csv_path = find_csv_file()
    raw = read_csv_safely(csv_path)

    if "행정구역" not in raw.columns:
        raise ValueError(
            "CSV 파일에서 '행정구역' 열을 찾을 수 없습니다."
        )

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
    raw["시군구"] = parent_info.apply(lambda value: value[1])

    total_population_columns = [
        column
        for column in raw.columns
        if re.search(r"_계_총인구수$", str(column))
    ]

    if not total_population_columns:
        raise ValueError(
            "'계_총인구수' 열을 찾을 수 없습니다."
        )

    total_population_column = total_population_columns[0]
    raw["총인구수"] = clean_number(raw[total_population_column])

    total_age_columns = {}
    male_age_columns = {}
    female_age_columns = {}

    for column in raw.columns:
        column_text = str(column)

        total_match = re.search(
            r"_계_(\d+)세$",
            column_text,
        )
        male_match = re.search(
            r"_남_(\d+)세$",
            column_text,
        )
        female_match = re.search(
            r"_여_(\d+)세$",
            column_text,
        )

        if total_match:
            total_age_columns[int(total_match.group(1))] = column
        elif male_match:
            male_age_columns[int(male_match.group(1))] = column
        elif female_match:
            female_age_columns[int(female_match.group(1))] = column

        if "_계_100세 이상" in column_text:
            total_age_columns[100] = column
        elif "_남_100세 이상" in column_text:
            male_age_columns[100] = column
        elif "_여_100세 이상" in column_text:
            female_age_columns[100] = column

    if not total_age_columns:
        raise ValueError(
            "연령별 전체 인구 열을 찾을 수 없습니다."
        )

    for age, column in total_age_columns.items():
        raw[f"전체_{age}세"] = clean_number(raw[column])

    for age, column in male_age_columns.items():
        raw[f"남자_{age}세"] = clean_number(raw[column])

    for age, column in female_age_columns.items():
        raw[f"여자_{age}세"] = clean_number(raw[column])

    raw = raw[
        raw["지역명"].notna()
        & raw["지역명"].ne("")
        & raw["총인구수"].gt(0)
    ].copy()

    raw = raw.drop_duplicates(
        subset=["행정구역코드", "지역명"],
        keep="first",
    ).reset_index(drop=True)

    return (
        raw,
        sorted(total_age_columns.keys()),
        male_age_columns,
        female_age_columns,
        csv_path,
    )


try:
    (
        population_df,
        available_ages,
        male_columns,
        female_columns,
        loaded_csv_path,
    ) = load_and_prepare_data()

except Exception as error:
    st.error("데이터를 불러오는 과정에서 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


# =========================================================
# 제목
# =========================================================
st.markdown(
    '<div class="main-title">👥 대한민국 쌍둥이 지역 찾기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        궁금한 지역을 선택하면 연령별 인구 구조가 가장 닮은 지역을 찾아줍니다.
        인구수 자체보다 각 연령층이 차지하는 비율을 중심으로 비교합니다.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.header("🔍 분석 조건")

    selected_level = st.radio(
        "분석할 행정구역 단계",
        options=["시도", "시군구", "읍면동"],
        horizontal=True,
        help=(
            "선택한 지역과 동일한 행정단계의 지역끼리 비교하는 것이 "
            "일반적으로 가장 정확합니다."
        ),
    )

    level_df = population_df[
        population_df["행정단계"] == selected_level
    ].copy()

    sido_options = ["전국"] + sorted(
        level_df["시도"].dropna().unique().tolist()
    )

    selected_sido_filter = st.selectbox(
        "지역 선택 범위",
        options=sido_options,
        help="선택 목록을 특정 시도로 좁힐 수 있습니다.",
    )

    selection_df = level_df.copy()

    if selected_sido_filter != "전국":
        selection_df = selection_df[
            selection_df["시도"] == selected_sido_filter
        ]

    selection_df = selection_df.sort_values("지역명")

    if selection_df.empty:
        st.warning("선택 조건에 해당하는 지역이 없습니다.")
        st.stop()

    selected_region_name = st.selectbox(
        "궁금한 지역",
        options=selection_df["지역명"].tolist(),
        index=0,
    )

    st.divider()

    st.subheader("⚙️ 유사도 설정")

    age_interval = st.select_slider(
        "연령구간 묶음",
        options=[1, 5, 10],
        value=5,
        format_func=lambda value: f"{value}세 단위",
        help=(
            "1세 단위는 매우 세밀하고, 5세 단위는 균형이 좋으며, "
            "10세 단위는 전체적인 구조를 비교하기 좋습니다."
        ),
    )

    population_weight = st.slider(
        "인구 규모 반영 비율",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        format="%d%%",
        help=(
            "0%이면 연령 비율만 비교합니다. 값을 높이면 전체 인구수까지 "
            "비슷한 지역이 높은 순위를 받습니다."
        ),
    )

    comparison_scope = st.radio(
        "쌍둥이 지역 검색 범위",
        options=[
            "전국 같은 행정단계",
            "같은 시도 안에서",
        ],
        index=0,
    )

    top_n = st.slider(
        "표시할 쌍둥이 지역 수",
        min_value=3,
        max_value=20,
        value=10,
    )

    exclude_same_sigungu = False

    if selected_level == "읍면동":
        exclude_same_sigungu = st.checkbox(
            "같은 시군구의 읍면동 제외",
            value=False,
            help=(
                "체크하면 선택 지역과 같은 시군구에 있는 동·읍·면은 "
                "검색 결과에서 제외됩니다."
            ),
        )

    st.divider()

    with st.expander("📁 데이터 정보"):
        st.write(f"파일명: `{loaded_csv_path.name}`")
        st.write(f"전체 행정구역 수: `{len(population_df):,}개`")
        st.write(
            f"현재 단계 지역 수: `{len(level_df):,}개`"
        )


# =========================================================
# 연령구간별 데이터 생성
# =========================================================
age_groups = {}

for age in available_ages:
    group_name = make_age_group(age, age_interval)
    age_groups.setdefault(group_name, []).append(age)

age_group_names = sorted(
    age_groups.keys(),
    key=age_sort_key,
)

for group_name, ages in age_groups.items():
    source_columns = [
        f"전체_{age}세"
        for age in ages
        if f"전체_{age}세" in population_df.columns
    ]

    population_df[f"연령그룹_{group_name}"] = population_df[
        source_columns
    ].sum(axis=1)

    population_df[f"비율_{group_name}"] = (
        population_df[f"연령그룹_{group_name}"]
        / population_df["총인구수"]
    )


feature_columns = [
    f"비율_{group_name}"
    for group_name in age_group_names
]


# =========================================================
# 선택 지역 및 비교 후보 구성
# =========================================================
selected_row = population_df[
    population_df["지역명"] == selected_region_name
].iloc[0]

comparison_df = level_df.copy()

if comparison_scope == "같은 시도 안에서":
    comparison_df = comparison_df[
        comparison_df["시도"] == selected_row["시도"]
    ]

if exclude_same_sigungu:
    comparison_df = comparison_df[
        comparison_df["시군구"] != selected_row["시군구"]
    ]

comparison_df = comparison_df[
    comparison_df["지역명"] != selected_region_name
].copy()

if comparison_df.empty:
    st.warning(
        "현재 검색 조건에서는 비교 가능한 다른 지역이 없습니다. "
        "검색 범위를 넓혀주세요."
    )
    st.stop()


# =========================================================
# 유사도 계산
# =========================================================
candidate_matrix = (
    comparison_df[feature_columns]
    .fillna(0)
    .to_numpy(dtype=float)
)

target_vector = (
    selected_row[feature_columns]
    .fillna(0)
    .to_numpy(dtype=float)
)

structure_score = cosine_similarity_matrix(
    candidate_matrix,
    target_vector,
)

pop_score = population_similarity(
    comparison_df["총인구수"].to_numpy(),
    selected_row["총인구수"],
)

population_weight_ratio = population_weight / 100
structure_weight_ratio = 1 - population_weight_ratio

final_score = (
    structure_score * structure_weight_ratio
    + pop_score * population_weight_ratio
)

comparison_df["인구구조유사도"] = structure_score * 100
comparison_df["인구규모유사도"] = pop_score * 100
comparison_df["최종유사도"] = final_score * 100
comparison_df["인구차이율"] = comparison_df["총인구수"].apply(
    lambda value: percentage_change(
        value,
        selected_row["총인구수"],
    )
)

ranked_df = (
    comparison_df
    .sort_values("최종유사도", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)

twin_row = ranked_df.iloc[0]


# =========================================================
# 요약 지표 계산
# =========================================================
def age_range_total(row: pd.Series, start_age: int, end_age: int) -> float:
    columns = [
        f"전체_{age}세"
        for age in available_ages
        if start_age <= age <= end_age
        and f"전체_{age}세" in row.index
    ]

    return float(row[columns].sum())


def demographic_summary(row: pd.Series) -> dict:
    total = max(float(row["총인구수"]), 1)

    children = age_range_total(row, 0, 14)
    working = age_range_total(row, 15, 64)
    elderly = age_range_total(row, 65, 100)

    return {
        "유소년": children / total * 100,
        "생산연령": working / total * 100,
        "고령": elderly / total * 100,
        "고령화지수": elderly / max(children, 1) * 100,
    }


selected_summary = demographic_summary(selected_row)
twin_summary = demographic_summary(twin_row)


# =========================================================
# 핵심 결과
# =========================================================
st.subheader("🏆 가장 닮은 쌍둥이 지역")

hero_col1, hero_col2, hero_col3 = st.columns(
    [1.3, 1, 1]
)

with hero_col1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-text">선택한 지역</div>
            <div style="font-size:1.5rem; font-weight:900; margin:5px 0;">
                {selected_region_name}
            </div>
            <div class="small-text">
                총인구 {format_population(selected_row["총인구수"])}
            </div>
            <hr style="opacity:0.15;">
            <div class="small-text">가장 닮은 지역</div>
            <div style="font-size:1.65rem; font-weight:900; color:#5b61f6; margin:5px 0;">
                {twin_row["지역명"]}
            </div>
            <div class="small-text">
                총인구 {format_population(twin_row["총인구수"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_col2:
    st.metric(
        "최종 유사도",
        f"{twin_row['최종유사도']:.2f}점",
        help="인구구조 유사도와 인구 규모 유사도를 합산한 점수입니다.",
    )

    st.metric(
        "연령구조 유사도",
        f"{twin_row['인구구조유사도']:.2f}점",
    )

with hero_col3:
    population_difference = (
        twin_row["총인구수"] - selected_row["총인구수"]
    )

    st.metric(
        "쌍둥이 지역 인구",
        format_population(twin_row["총인구수"]),
        delta=f"{population_difference:+,.0f}명",
    )

    st.metric(
        "고령인구 비율 차이",
        f"{twin_summary['고령']:.1f}%",
        delta=(
            f"{twin_summary['고령'] - selected_summary['고령']:+.1f}%p"
        ),
        delta_color="off",
    )


# =========================================================
# 탭 구성
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 핵심 비교",
        "👨‍👩‍👧 인구 피라미드",
        "🗺️ 유사 지역 지도",
        "📋 전체 순위",
    ]
)


# =========================================================
# 탭 1: 핵심 비교
# =========================================================
with tab1:
    st.subheader("연령구조 비교")

    age_comparison_rows = []

    for group_name in age_group_names:
        selected_percentage = (
            selected_row[f"비율_{group_name}"] * 100
        )
        twin_percentage = (
            twin_row[f"비율_{group_name}"] * 100
        )

        age_comparison_rows.append(
            {
                "연령구간": group_name,
                "지역": selected_region_name,
                "인구비율": selected_percentage,
            }
        )

        age_comparison_rows.append(
            {
                "연령구간": group_name,
                "지역": twin_row["지역명"],
                "인구비율": twin_percentage,
            }
        )

    age_comparison_df = pd.DataFrame(age_comparison_rows)

    age_line_fig = px.line(
        age_comparison_df,
        x="연령구간",
        y="인구비율",
        color="지역",
        markers=True,
        custom_data=["지역"],
        title="연령구간별 인구 비율",
    )

    age_line_fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "연령구간: %{x}<br>"
            "인구비율: %{y:.2f}%"
            "<extra></extra>"
        ),
        line=dict(width=3),
        marker=dict(size=7),
    )

    age_line_fig.update_layout(
        height=500,
        hovermode="x unified",
        legend_title_text="",
        xaxis_title="연령구간",
        yaxis_title="전체 인구 중 비율(%)",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(
        age_line_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    radar_col, group_col = st.columns(2)

    with radar_col:
        radar_categories = [
            "유소년 인구",
            "생산연령 인구",
            "고령인구",
        ]

        radar_fig = go.Figure()

        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    selected_summary["유소년"],
                    selected_summary["생산연령"],
                    selected_summary["고령"],
                ],
                theta=radar_categories,
                fill="toself",
                name=selected_region_name,
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    twin_summary["유소년"],
                    twin_summary["생산연령"],
                    twin_summary["고령"],
                ],
                theta=radar_categories,
                fill="toself",
                name=twin_row["지역명"],
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            )
        )

        radar_fig.update_layout(
            title="3대 인구계층 비교",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    ticksuffix="%",
                )
            ),
            height=430,
            legend_title_text="",
            margin=dict(l=40, r=40, t=70, b=40),
        )

        st.plotly_chart(
            radar_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with group_col:
        major_group_rows = []

        for region_label, summary in [
            (selected_region_name, selected_summary),
            (twin_row["지역명"], twin_summary),
        ]:
            major_group_rows.extend(
                [
                    {
                        "지역": region_label,
                        "인구계층": "유소년 0~14세",
                        "비율": summary["유소년"],
                    },
                    {
                        "지역": region_label,
                        "인구계층": "생산연령 15~64세",
                        "비율": summary["생산연령"],
                    },
                    {
                        "지역": region_label,
                        "인구계층": "고령 65세 이상",
                        "비율": summary["고령"],
                    },
                ]
            )

        major_group_df = pd.DataFrame(major_group_rows)

        major_group_fig = px.bar(
            major_group_df,
            x="인구계층",
            y="비율",
            color="지역",
            barmode="group",
            text_auto=".1f",
            title="주요 인구계층 비율",
        )

        major_group_fig.update_traces(
            texttemplate="%{y:.1f}%",
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{x}<br>"
                "%{y:.2f}%"
                "<extra></extra>"
            ),
        )

        major_group_fig.update_layout(
            height=430,
            xaxis_title="",
            yaxis_title="비율(%)",
            legend_title_text="",
            margin=dict(l=20, r=20, t=70, b=30),
        )

        st.plotly_chart(
            major_group_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.subheader("두 지역의 차이가 큰 연령구간")

    difference_df = pd.DataFrame(
        {
            "연령구간": age_group_names,
            "선택지역": [
                selected_row[f"비율_{group}"] * 100
                for group in age_group_names
            ],
            "쌍둥이지역": [
                twin_row[f"비율_{group}"] * 100
                for group in age_group_names
            ],
        }
    )

    difference_df["비율차이"] = (
        difference_df["쌍둥이지역"]
        - difference_df["선택지역"]
    )

    difference_df["절대차이"] = (
        difference_df["비율차이"].abs()
    )

    difference_df = difference_df.sort_values(
        "절대차이",
        ascending=False,
    ).head(10)

    difference_fig = px.bar(
        difference_df.sort_values("비율차이"),
        x="비율차이",
        y="연령구간",
        orientation="h",
        text_auto=".2f",
        title=(
            f"{twin_row['지역명']}의 비율 - "
            f"{selected_region_name}의 비율"
        ),
    )

    difference_fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
    )

    difference_fig.update_traces(
        texttemplate="%{x:+.2f}%p",
        hovertemplate=(
            "연령구간: %{y}<br>"
            "비율 차이: %{x:+.3f}%p"
            "<extra></extra>"
        ),
    )

    difference_fig.update_layout(
        height=480,
        xaxis_title="인구 비율 차이(%p)",
        yaxis_title="",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(
        difference_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )


# =========================================================
# 탭 2: 인구 피라미드
# =========================================================
with tab2:
    st.subheader("성별·연령별 인구 피라미드")

    pyramid_region = st.radio(
        "피라미드에 표시할 지역",
        options=[
            selected_region_name,
            twin_row["지역명"],
        ],
        horizontal=True,
    )

    if pyramid_region == selected_region_name:
        pyramid_row = selected_row
    else:
        pyramid_row = twin_row

    pyramid_interval = st.select_slider(
        "피라미드 연령구간",
        options=[1, 5, 10],
        value=5,
        format_func=lambda value: f"{value}세 단위",
        key="pyramid_interval",
    )

    pyramid_groups = {}

    for age in available_ages:
        group_name = make_age_group(age, pyramid_interval)
        pyramid_groups.setdefault(group_name, []).append(age)

    pyramid_rows = []

    for group_name, ages in pyramid_groups.items():
        male_total = 0
        female_total = 0

        for age in ages:
            male_column = f"남자_{age}세"
            female_column = f"여자_{age}세"

            if male_column in pyramid_row.index:
                male_total += float(pyramid_row[male_column])

            if female_column in pyramid_row.index:
                female_total += float(pyramid_row[female_column])

        pyramid_rows.append(
            {
                "연령구간": group_name,
                "남자": -male_total,
                "여자": female_total,
            }
        )

    pyramid_df = pd.DataFrame(pyramid_rows)
    pyramid_df = pyramid_df.sort_values(
        "연령구간",
        key=lambda series: series.map(age_sort_key),
    )

    pyramid_fig = go.Figure()

    pyramid_fig.add_trace(
        go.Bar(
            y=pyramid_df["연령구간"],
            x=pyramid_df["남자"],
            name="남자",
            orientation="h",
            customdata=np.abs(pyramid_df["남자"]),
            hovertemplate=(
                "남자<br>"
                "%{y}<br>"
                "%{customdata:,.0f}명"
                "<extra></extra>"
            ),
        )
    )

    pyramid_fig.add_trace(
        go.Bar(
            y=pyramid_df["연령구간"],
            x=pyramid_df["여자"],
            name="여자",
            orientation="h",
            hovertemplate=(
                "여자<br>"
                "%{y}<br>"
                "%{x:,.0f}명"
                "<extra></extra>"
            ),
        )
    )

    maximum_population = max(
        abs(pyramid_df["남자"]).max(),
        pyramid_df["여자"].max(),
    )

    tick_values = np.linspace(
        -maximum_population,
        maximum_population,
        9,
    )

    pyramid_fig.update_layout(
        title=f"{pyramid_region} 인구 피라미드",
        barmode="relative",
        height=max(550, len(pyramid_df) * 27),
        xaxis=dict(
            title="인구수",
            tickvals=tick_values,
            ticktext=[
                f"{abs(value):,.0f}"
                for value in tick_values
            ],
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=pyramid_df["연령구간"].tolist(),
        ),
        legend_title_text="",
        bargap=0.12,
        margin=dict(l=20, r=20, t=70, b=30),
    )

    st.plotly_chart(
        pyramid_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.caption(
        "남자는 왼쪽, 여자는 오른쪽에 표시됩니다. "
        "마우스를 올리면 실제 인구수를 확인할 수 있습니다."
    )


# =========================================================
# 탭 3: PCA 유사 지역 분포
# =========================================================
with tab3:
    st.subheader("인구구조 공간에서 지역들의 위치")

    st.caption(
        "연령별 인구 비율이 비슷한 지역일수록 그래프에서 가까운 위치에 표시됩니다. "
        "PCA는 여러 연령 비율을 2차원으로 압축한 결과입니다."
    )

    pca_df = level_df.copy()
    pca_matrix = (
        pca_df[feature_columns]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    feature_mean = pca_matrix.mean(axis=0)
    feature_std = pca_matrix.std(axis=0)
    feature_std = np.where(feature_std == 0, 1, feature_std)

    standardized_matrix = (
        pca_matrix - feature_mean
    ) / feature_std

    centered_matrix = (
        standardized_matrix
        - standardized_matrix.mean(axis=0)
    )

    _, singular_values, right_vectors = np.linalg.svd(
        centered_matrix,
        full_matrices=False,
    )

    coordinates = np.dot(
        centered_matrix,
        right_vectors[:2].T,
    )

    pca_df["PCA1"] = coordinates[:, 0]
    pca_df["PCA2"] = coordinates[:, 1]

    total_variance = np.sum(singular_values ** 2)

    if total_variance > 0:
        explained_variance = (
            singular_values[:2] ** 2
        ) / total_variance * 100
    else:
        explained_variance = np.array([0, 0])

    highlighted_regions = (
        [selected_region_name]
        + ranked_df["지역명"].head(5).tolist()
    )

    pca_df["표시구분"] = np.where(
        pca_df["지역명"] == selected_region_name,
        "선택 지역",
        np.where(
            pca_df["지역명"] == twin_row["지역명"],
            "1위 쌍둥이",
            np.where(
                pca_df["지역명"].isin(highlighted_regions),
                "상위 유사 지역",
                "기타 지역",
            ),
        ),
    )

    pca_df["마커크기"] = np.where(
        pca_df["표시구분"] == "기타 지역",
        5,
        14,
    )

    pca_fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="표시구분",
        size="마커크기",
        size_max=18,
        hover_name="지역명",
        hover_data={
            "시도": True,
            "총인구수": ":,.0f",
            "PCA1": ":.2f",
            "PCA2": ":.2f",
            "마커크기": False,
        },
        category_orders={
            "표시구분": [
                "기타 지역",
                "상위 유사 지역",
                "1위 쌍둥이",
                "선택 지역",
            ]
        },
        title="전국 지역 인구구조 PCA 분포",
    )

    pca_fig.update_traces(
        marker=dict(opacity=0.75),
    )

    pca_fig.update_layout(
        height=650,
        xaxis_title=(
            f"인구구조 주성분 1 "
            f"({explained_variance[0]:.1f}% 설명)"
        ),
        yaxis_title=(
            f"인구구조 주성분 2 "
            f"({explained_variance[1]:.1f}% 설명)"
        ),
        legend_title_text="",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(
        pca_fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    st.info(
        "이 그래프의 거리는 시각화를 위해 단순화된 거리이며, "
        "실제 순위는 전체 연령구간을 사용한 유사도 점수로 계산합니다."
    )


# =========================================================
# 탭 4: 전체 순위
# =========================================================
with tab4:
    st.subheader(f"{selected_region_name}의 쌍둥이 지역 순위")

    rank_chart_df = ranked_df.sort_values(
        "최종유사도",
        ascending=True,
    )

    rank_fig = px.bar(
        rank_chart_df,
        x="최종유사도",
        y="지역명",
        orientation="h",
        text_auto=".2f",
        custom_data=[
            "인구구조유사도",
            "인구규모유사도",
            "총인구수",
        ],
        title=f"상위 {len(ranked_df)}개 지역 유사도",
    )

    rank_fig.update_traces(
        texttemplate="%{x:.2f}",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "최종 유사도: %{x:.2f}점<br>"
            "인구구조 유사도: %{customdata[0]:.2f}점<br>"
            "인구규모 유사도: %{customdata[1]:.2f}점<br>"
            "총인구: %{customdata[2]:,.0f}명"
            "<extra></extra>"
        ),
    )

    minimum_score = max(
        0,
        ranked_df["최종유사도"].min() - 2,
    )

    rank_fig.update_layout(
        height=max(450, len(ranked_df) * 42),
        xaxis=dict(
            title="최종 유사도 점수",
            range=[minimum_score, 100],
        ),
        yaxis_title="",
        margin=dict(l=20, r=20, t=70, b=20),
    )

    st.plotly_chart(
        rank_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.subheader("상세 순위표")

    display_rank_df = ranked_df.copy()
    display_rank_df.insert(
        0,
        "순위",
        np.arange(1, len(display_rank_df) + 1),
    )

    display_rank_df["총인구"] = display_rank_df[
        "총인구수"
    ].apply(format_population)

    display_rank_df["최종 유사도"] = display_rank_df[
        "최종유사도"
    ].map(lambda value: f"{value:.2f}")

    display_rank_df["인구구조 유사도"] = display_rank_df[
        "인구구조유사도"
    ].map(lambda value: f"{value:.2f}")

    display_rank_df["인구 규모 유사도"] = display_rank_df[
        "인구규모유사도"
    ].map(lambda value: f"{value:.2f}")

    display_rank_df["선택 지역 대비 인구 차이"] = display_rank_df[
        "인구차이율"
    ].map(lambda value: f"{value:+.1f}%")

    st.dataframe(
        display_rank_df[
            [
                "순위",
                "지역명",
                "시도",
                "총인구",
                "최종 유사도",
                "인구구조 유사도",
                "인구 규모 유사도",
                "선택 지역 대비 인구 차이",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "순위": st.column_config.NumberColumn(
                width="small",
            ),
            "지역명": st.column_config.TextColumn(
                width="large",
            ),
        },
    )

    csv_download = display_rank_df[
        [
            "순위",
            "지역명",
            "시도",
            "행정단계",
            "총인구",
            "최종 유사도",
            "인구구조 유사도",
            "인구 규모 유사도",
            "선택 지역 대비 인구 차이",
        ]
    ].to_csv(
        index=False,
        encoding="utf-8-sig",
    )

    st.download_button(
        label="📥 쌍둥이 지역 순위 CSV 다운로드",
        data=csv_download,
        file_name=f"{selected_region_name}_쌍둥이지역.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# 분석 방법 안내
# =========================================================
st.divider()

with st.expander("🧮 쌍둥이 지역은 어떻게 계산하나요?"):
    st.markdown(
        f"""
        ### 1. 연령별 인구 비율 계산

        각 지역의 전체 인구에서 각 연령구간이 차지하는 비율을 계산합니다.

        현재 설정은 **{age_interval}세 단위 연령구간**입니다.

        ### 2. 인구구조 유사도 계산

        선택 지역과 비교 지역의 연령 비율 벡터에 코사인 유사도를 적용합니다.

        - 연령 분포의 모양이 같을수록 100점에 가까워집니다.
        - 전체 인구가 크게 달라도 연령 비율이 같으면 높은 점수를 받을 수 있습니다.

        ### 3. 인구 규모 유사도 계산

        두 지역의 총인구 비율을 로그 척도로 비교합니다.

        현재 인구 규모 반영 비율은 **{population_weight}%**입니다.

        ### 4. 최종 점수

        최종 유사도는 다음 두 점수를 가중평균한 결과입니다.

        - 인구구조 유사도: **{100 - population_weight}%**
        - 인구 규모 유사도: **{population_weight}%**

        인구 규모 반영 비율을 0%로 설정하면 순수하게 연령구조만 비교합니다.
        """
    )

st.caption(
    "데이터 출처: 업로드된 주민등록 연령별 인구현황 CSV · "
    "분석 결과는 행정·정책 결정을 위한 공식 통계 분석을 대체하지 않습니다."
)
