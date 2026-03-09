import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import sweetviz as sv
from ydata_profiling import ProfileReport

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="통신 세그먼트 분석 자동화", layout="wide")

# --- 2. 데이터 생성 (사업팀 요청 시나리오 반영) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_rows = 500
    data = {
        '고객ID': range(10001, 10001 + n_rows),
        '약정유형': np.random.choice(['SIM-only', '단말약정'], n_rows),
        '요금제레벨': np.random.choice(['고가', '중저가'], n_rows),
        '단말유형': np.random.choice(['iPhone', '갤럭시 프리미엄', '갤럭시 중저가', '기타'], n_rows),
        '나이': np.random.randint(18, 70, n_rows),
        '데이터사용량_GB': np.random.uniform(5, 150, n_rows),
        '월평균매출_ARPU': np.random.uniform(30000, 100000, n_rows),
        '이탈여부': np.random.choice([0, 1], n_rows, p=[0.85, 0.15])
    }
    # 시나리오 기반 상관관계 주입 (SIM-only가 데이터 사용량이 더 많도록 설정)
    df = pd.DataFrame(data)
    df.loc[df['약정유형'] == 'SIM-only', '데이터사용량_GB'] += 20
    return df

# --- 3. 메인 로직 ---
def main():
    st.title("📊 통신 고객 세그먼트 비교 분석 자동화")
    st.markdown("사업팀 요청 EDA 및 분석가용 기술 통계를 자동으로 생성합니다.")

    df = load_data()

    # 사이드바: 분석 그룹 설정
    st.sidebar.header("🔍 비교 세그먼트 설정")
    
    # 분석 차원 선택
    dimension = st.sidebar.selectbox("비교할 디멘젼 선택:", ["약정유형", "요금제레벨", "단말유형"])
    
    # 그룹 A/B 필터링
    unique_vals = df[dimension].unique().tolist()
    group_a_val = st.sidebar.selectbox(f"그룹 A ({dimension}):", unique_vals, index=0)
    group_b_val = st.sidebar.selectbox(f"그룹 B ({dimension}):", unique_vals, index=min(1, len(unique_vals)-1))

    st.sidebar.divider()
    api_key = st.sidebar.text_input("Gemini API Key (선택)", type="password")
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["🎯 세그먼트 비교 (Sweetviz)", "📈 데이터 상세 (YData)", "🛠️ 변수 최적화 (OptBinning)"])

    group_a = df[df[dimension] == group_a_val]
    group_b = df[df[dimension] == group_b_val]

    with tab1:
        st.subheader(f"✅ {group_a_val} vs {group_b_val} 특성 비교")
        st.write("사업팀이 정의한 두 그룹의 모든 디멘젼 차이를 자동으로 분석합니다.")
        
        if st.button("실시간 비교 리포트 생성"):
            with st.spinner("Sweetviz가 두 그룹의 차이를 계산 중입니다..."):
                # Sweetviz 비교 분석 실행
                report = sv.compare([group_a, group_a_val], [group_b, group_b_val], target_feat='이탈여부')
                report.show_html(filepath='compare_report.html
