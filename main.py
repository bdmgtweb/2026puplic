from pathlib import Path
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
    raw["시군구"] = parent_info.apply(lambda value: value[1])

    total_population_columns = [
        column
        for column in raw.columns
        if re.search(r"_계_총인구수$", str(column))
    ]

    if not total_population_columns:
        total_population_columns = [
            column
            for column in raw.columns
            if "총인구수" in str(column) and "_계_" in str(column)
        ]

    if not total_population_columns:
        raise ValueError("'계_총인구수' 열을 찾을 수 없습니다.")

    raw["총인구수"] = clean_number(raw[total_population_columns[0]])

    total_age_columns = {}
    male_age_columns = {}
    female_age_columns = {}

    for column in raw.columns:
        column_text = str(column).strip()

        total_match = re.search(r"_계_(\d+)세$", column_text)
        male_match = re.search(r"_남_(\d+)세$", column_text)
        female_match = re.search(r"_여_(\d+)세$", column_text)

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
        raise ValueError("연령별 전체 인구 열을 찾을 수 없습니다.")

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
        csv_path.name,
    )


try:
    population_df, available_ages, loaded_csv_name = load_and_prepare_data()
except Exception as error:
    st.error("데이터를 불러오는 과정에서 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


st.markdown(
    '<div class="main-title">👥 대한민국 쌍둥이 지역 찾기</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        궁금한 지역을 선택하면 연령별 인구구조가 가장 닮은 지역을 찾아줍니다.
        인구수 자체보다 각 연령층이 차지하는 비율을 중심으로 비교합니다.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("🔍 분석 조건")

    selected_level = st.radio(
        "분석할 행정구역 단계",
        options=["시도", "시군구", "읍면동"],
        horizontal=True,
    )

    age_interval = st.select_slider(
        "연령구간 묶음",
        options=[1, 5, 10],
        value=5,
        format_func=lambda value: f"{value}세 단위",
    )


age_groups = {}

for age in available_ages:
    group_name = make_age_group(age, age_interval)
    age_groups.setdefault(group_name, []).append(age)

age_group_names = sorted(age_groups.keys(), key=age_sort_key)

for group_name, ages in age_groups.items():
    source_columns = [
        f"전체_{age}세"
        for age in ages
        if f"전체_{age}세" in population_df.columns
    ]

    group_column = f"연령그룹_{group_name}"
    ratio_column = f"비율_{group_name}"

    if source_columns:
        population_df[group_column] = population_df[source_columns].sum(axis=1)
    else:
        population_df[group_column] = 0

    population_df[ratio_column] = np.where(
        population_df["총인구수"] > 0,
        population_df[group_column] / population_df["총인구수"],
        0,
    )


feature_columns = [f"비율_{group_name}" for group_name in age_group_names]

for column in feature_columns:
    if column not in population_df.columns:
        population_df[column] = 0


level_df = population_df[
    population_df["행정단계"] == selected_level
].copy()

if level_df.empty:
    st.warning(f"'{selected_level}' 단계의 지역 데이터가 없습니다.")
    st.stop()


with st.sidebar:
    sido_options = ["전국"] + sorted(
        level_df["시도"].dropna().astype(str).unique().tolist()
    )

    selected_sido_filter = st.selectbox(
        "지역 선택 범위",
        options=sido_options,
    )

    selection_df = level_df.copy()

    if selected_sido_filter != "전국":
        selection_df = selection_df[
            selection_df["시도"] == selected_sido_filter
        ].copy()

    selection_df = selection_df.sort_values("지역명")

    if selection_df.empty:
        st.warning("선택 조건에 해당하는 지역이 없습니다.")
        st.stop()

    selected_region_name = st.selectbox(
        "궁금한 지역",
        options=selection_df["지역명"].tolist(),
    )

    st.divider()
    st.subheader("⚙️ 유사도 설정")

    population_weight = st.slider(
        "인구 규모 반영 비율",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        format="%d%%",
    )

    comparison_scope = st.radio(
        "쌍둥이 지역 검색 범위",
        options=["전국 같은 행정단계", "같은 시도 안에서"],
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
        )

    with st.expander("📁 데이터 정보"):
        st.write(f"파일명: `{loaded_csv_name}`")
        st.write(f"전체 행정구역 수: `{len(population_df):,}개`")
        st.write(f"현재 단계 지역 수: `{len(level_df):,}개`")


selected_rows = population_df[
    population_df["지역명"] == selected_region_name
].copy()

if selected_rows.empty:
    st.error("선택한 지역의 데이터를 찾을 수 없습니다.")
    st.stop()

selected_row = selected_rows.iloc[0].copy()


comparison_df = level_df.copy()

if comparison_scope == "같은 시도 안에서":
    comparison_df = comparison_df[
        comparison_df["시도"] == selected_row["시도"]
    ].copy()

if exclude_same_sigungu and selected_level == "읍면동":
    comparison_df = comparison_df[
        comparison_df["시군구"] != selected_row["시군구"]
    ].copy()

comparison_df = comparison_df[
    comparison_df["지역명"] != selected_region_name
].copy()

if comparison_df.empty:
    st.warning("현재 조건에서는 비교할 수 있는 다른 지역이 없습니다.")
    st.stop()


for column in feature_columns:
    if column not in comparison_df.columns:
        comparison_df[column] = 0

    if column not in selected_row.index:
        selected_row.loc[column] = 0


comparison_df[feature_columns] = (
    comparison_df[feature_columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)

selected_feature_series = pd.to_numeric(
    selected_row[feature_columns],
    errors="coerce",
).fillna(0)


candidate_matrix = comparison_df[feature_columns].to_numpy(dtype=float)
target_vector = selected_feature_series.to_numpy(dtype=float)

structure_score = cosine_similarity_matrix(
    candidate_matrix,
    target_vector,
)

pop_score = population_similarity(
    comparison_df["총인구수"].to_numpy(dtype=float),
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

if ranked_df.empty:
    st.warning("쌍둥이 지역 분석 결과가 없습니다.")
    st.stop()

twin_row = ranked_df.iloc[0].copy()

selected_summary = demographic_summary(
    selected_row,
    available_ages,
)

twin_summary = demographic_summary(
    twin_row,
    available_ages,
)


st.subheader("🏆 가장 닮은 쌍둥이 지역")

hero_col1, hero_col2, hero_col3 = st.columns([1.3, 1, 1])

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
    )

    st.metric(
        "연령구조 유사도",
        f"{twin_row['인구구조유사도']:.2f}점",
    )

with hero_col3:
    population_difference = (
        float(twin_row["총인구수"])
        - float(selected_row["총인구수"])
    )

    elderly_ratio_difference = (
        twin_summary["고령"]
        - selected_summary["고령"]
    )

    st.metric(
        "쌍둥이 지역 인구",
        format_population(twin_row["총인구수"]),
        delta=f"{population_difference:+,.0f}명",
    )

    st.metric(
        "고령인구 비율",
        f"{twin_summary['고령']:.1f}%",
        delta=f"{elderly_ratio_difference:+.1f}%p",
        delta_color="off",
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 핵심 비교",
        "👨‍👩‍👧 인구 피라미드",
        "🗺️ 유사 지역 분포",
        "📋 전체 순위",
    ]
)


with tab1:
    st.subheader("연령구조 비교")

    age_comparison_rows = []

    for group_name in age_group_names:
        ratio_column = f"비율_{group_name}"

        selected_percentage = float(
            selected_row.get(ratio_column, 0)
        ) * 100

        twin_percentage = float(
            twin_row.get(ratio_column, 0)
        ) * 100

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
        title="연령구간별 인구 비율",
    )

    age_line_fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "연령구간: %{x}<br>"
            "인구비율: %{y:.2f}%"
            "<extra></extra>"
        ),
    )

    age_line_fig.update_layout(
        height=500,
        hovermode="x unified",
        legend_title_text="",
        xaxis_title="연령구간",
        yaxis_title="전체 인구 중 비율(%)",
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

        major_group_fig.update_layout(
            height=430,
            xaxis_title="",
            yaxis_title="비율(%)",
            legend_title_text="",
        )

        st.plotly_chart(
            major_group_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.subheader("두 지역의 차이가 큰 연령구간")

    difference_rows = []

    for group_name in age_group_names:
        ratio_column = f"비율_{group_name}"

        selected_value = float(
            selected_row.get(ratio_column, 0)
        ) * 100

        twin_value = float(
            twin_row.get(ratio_column, 0)
        ) * 100

        difference_rows.append(
            {
                "연령구간": group_name,
                "비율차이": twin_value - selected_value,
            }
        )

    difference_df = pd.DataFrame(difference_rows)
    difference_df["절대차이"] = difference_df["비율차이"].abs()

    difference_df = (
        difference_df
        .sort_values("절대차이", ascending=False)
        .head(10)
    )

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

    difference_fig.update_layout(
        height=480,
        xaxis_title="인구 비율 차이(%p)",
        yaxis_title="",
    )

    st.plotly_chart(
        difference_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )


with tab2:
    st.subheader("성별·연령별 인구 피라미드")

    pyramid_region = st.radio(
        "피라미드에 표시할 지역",
        options=[selected_region_name, twin_row["지역명"]],
        horizontal=True,
    )

    pyramid_row = (
        selected_row
        if pyramid_region == selected_region_name
        else twin_row
    )

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
            male_total += float(
                pyramid_row.get(f"남자_{age}세", 0)
            )

            female_total += float(
                pyramid_row.get(f"여자_{age}세", 0)
            )

        pyramid_rows.append(
            {
                "연령구간": group_name,
                "남자": -male_total,
                "여자": female_total,
            }
        )

    pyramid_df = pd.DataFrame(pyramid_rows)
    pyramid_df["정렬값"] = pyramid_df["연령구간"].map(age_sort_key)
    pyramid_df = pyramid_df.sort_values("정렬값")

    pyramid_fig = go.Figure()

    pyramid_fig.add_trace(
        go.Bar(
            y=pyramid_df["연령구간"],
            x=pyramid_df["남자"],
            name="남자",
            orientation="h",
            customdata=np.abs(pyramid_df["남자"]),
            hovertemplate=(
                "남자<br>%{y}<br>%{customdata:,.0f}명"
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
                "여자<br>%{y}<br>%{x:,.0f}명"
                "<extra></extra>"
            ),
        )
    )

    maximum_population = max(
        float(np.abs(pyramid_df["남자"]).max()),
        float(pyramid_df["여자"].max()),
        1,
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
    )

    st.plotly_chart(
        pyramid_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )


with tab3:
    st.subheader("인구구조 공간에서 지역들의 위치")

    pca_df = level_df.copy()

    pca_df[feature_columns] = (
        pca_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    pca_matrix = pca_df[feature_columns].to_numpy(dtype=float)

    if pca_matrix.shape[0] < 2 or pca_matrix.shape[1] < 2:
        st.info(
            "PCA 그래프를 표시하려면 최소 2개 이상의 지역과 "
            "연령구간이 필요합니다."
        )
    else:
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

        try:
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
                    / total_variance
                    * 100
                )
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
                title="전국 지역 인구구조 PCA 분포",
            )

            pca_fig.update_layout(
                height=650,
                xaxis_title=(
                    "인구구조 주성분 1 "
                    f"({explained_variance[0]:.1f}% 설명)"
                ),
                yaxis_title=(
                    "인구구조 주성분 2 "
                    f"({explained_variance[1]:.1f}% 설명)"
                ),
                legend_title_text="",
            )

            st.plotly_chart(
                pca_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

        except Exception as error:
            st.warning("PCA 그래프를 생성하지 못했습니다.")
            st.caption(str(error))


with tab4:
    st.subheader(
        f"{selected_region_name}의 쌍둥이 지역 순위"
    )

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
        float(ranked_df["최종유사도"].min()) - 2,
    )

    rank_fig.update_layout(
        height=max(450, len(ranked_df) * 42),
        xaxis=dict(
            title="최종 유사도 점수",
            range=[minimum_score, 100],
        ),
        yaxis_title="",
    )

    st.plotly_chart(
        rank_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    display_rank_df = ranked_df.copy()

    display_rank_df.insert(
        0,
        "순위",
        np.arange(1, len(display_rank_df) + 1),
    )

    display_rank_df["총인구"] = (
        display_rank_df["총인구수"]
        .apply(format_population)
    )

    display_rank_df["최종 유사도"] = (
        display_rank_df["최종유사도"]
        .map(lambda value: f"{value:.2f}")
    )

    display_rank_df["인구구조 유사도"] = (
        display_rank_df["인구구조유사도"]
        .map(lambda value: f"{value:.2f}")
    )

    display_rank_df["인구 규모 유사도"] = (
        display_rank_df["인구규모유사도"]
        .map(lambda value: f"{value:.2f}")
    )

    display_rank_df["선택 지역 대비 인구 차이"] = (
        display_rank_df["인구차이율"]
        .map(lambda value: f"{value:+.1f}%")
    )

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

    safe_file_name = re.sub(
        r'[\\/:*?"<>|]',
        "_",
        selected_region_name,
    )

    st.download_button(
        label="📥 쌍둥이 지역 순위 CSV 다운로드",
        data=csv_download,
        file_name=f"{safe_file_name}_쌍둥이지역.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()

with st.expander("🧮 쌍둥이 지역은 어떻게 계산하나요?"):
    st.markdown(
        f"""
        ### 1. 연령별 인구 비율 계산

        현재 설정은 **{age_interval}세 단위 연령구간**입니다.

        ### 2. 인구구조 유사도 계산

        선택 지역과 비교 지역의 연령 비율 벡터에
        코사인 유사도를 적용합니다.

        ### 3. 인구 규모 반영

        현재 인구 규모 반영 비율은
        **{population_weight}%**입니다.

        - 인구구조 유사도: **{100 - population_weight}%**
        - 인구 규모 유사도: **{population_weight}%**
        """
    )

st.caption(
    "데이터 출처: 업로드된 주민등록 연령별 인구현황 CSV"
)
