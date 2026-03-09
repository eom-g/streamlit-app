import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import os

# --- 1. 라이브러리 로드 가드 (버전 충돌 방지) ---
try:
    import sweetviz as sv
except Exception:
    sv = None

try:
    from ydata_profiling import ProfileReport
except Exception:
    ProfileReport = None

# --- 2. 데이터 생성 로직 ---
@st.cache_data
def load_telco_data():
    np.random.seed(42)
    n_rows = 200
    data = {
        '고객ID': range(1001, 1001 + n_rows),
        '나이': np.random.randint(18, 75, n_rows),
        '접속_온라인쇼핑': np.random.randint(5, 100, n_rows),
        '접속_음악스트리밍': np.random.randint(10, 200, n_rows),
        '접속_OTT영상': np.random.randint(5, 150, n_rows),
        '월_데이터사용량_GB': np.random.uniform(2, 120, n_rows),
        '단말사용기간_개월': np.random.randint(1, 60, n_rows),
        '결합유형': np.random.choice(['미결합', '유무선결합', '무무선결합'], n_rows),
        '이탈여부': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# --- 3. 가설 관련 변수 필터링 ---
def get_related_cols(hypothesis):
    mapping = {
        'OTT': ['접속_OTT영상', '월_데이터사용량_GB'],
        '데이터': ['월_데이터사용량_GB'],
        '쇼핑': ['접속_온라인쇼핑'],
        '나이': ['나이'],
        '결합': ['결합유형']
    }
    cols = ['고객ID', '이탈여부']
    for key, val in mapping.items():
        if key in hypothesis:
            cols.extend(val)
    return list(set(cols))

# --- 4. 메인 UI 구성 ---
def main():
    st.set_page_config(page_title="Awesome EDA Sandbox", layout="wide")
    
    st.title("🧪 통신 고객 가설 검증 & Awesome EDA")
    df = load_telco_data()

    # 사이드바
    with st.sidebar:
        st.header("1. 가설 설정")
        user_hypo = st.text_area("검증할 가설을 입력하세요:", 
                                value="OTT 영상 접속이 많은 고객은 데이터 사용량도 많을 것이다.")
        api_key = st.text_input("Gemini API Key", type="password")
        run_btn = st.button("가설 분석 시작", use_container_width=True)

    if run_btn:
        rel_cols = get_related_cols(user_hypo)
        if len(rel_cols) <= 2: rel_cols = df.columns.tolist()
        target_df = df[rel_cols]

        st.success(f"✔️ 분석 대상 변수: {', '.join([c for c in rel_cols if c != '고객ID'])}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 현황", "🍭 Sweetviz", "✨ YData Report", "📈 OptBinning"])

        with tab1:
            st.subheader("추출 데이터 샘플")
            st.dataframe(target_df.head(10))
            st.write("변수 간 상관계수")
            st.dataframe(target_df.select_dtypes(include=[np.number]).corr())

        with tab2:
            st.subheader("Sweetviz 비교 분석")
            if sv is not None:
                with st.spinner("리포트 생성 중..."):
                    report = sv.analyze(target_df, target_feat='이탈여부')
                    report.show_html(filepath='sv_report.html', open_browser=False)
                    with open('sv_report.html', 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=900, scrolling=True)
            else:
                st.error("Sweetviz 라이브러리를 현재 환경에서 사용할 수 없습니다.")

        with tab3:
            st.subheader("YData Profiling 상세 통계")
            if ProfileReport is not None:
                with st.spinner("프로파일링 생성 중..."):
                    pr = ProfileReport(target_df, explorative=True, minimal=True)
                    components.html(pr.to_html(), height=900, scrolling=True)
            else:
                st.error("YData-Profiling 라이브러리를 로드할 수 없습니다.")

        with tab4:
            st.subheader("Optimal Binning 분석")
            num_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c not in ['고객ID', '이탈여부']]
            
            if num_cols:
                sel_col = st.selectbox("분석 변수 선택:", num_cols)
                optb = OptimalBinning(name=sel_col, dtype="numerical", solver="cp")
                optb.fit(target_df[sel_col].values, target_df['이탈여부'].values)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Binning Table**")
                    st.dataframe(optb.binning_table.build())
                with col2:
                    st.write("**WoE 차트**")
                    bin_df = optb.binning_table.build()[:-1]
                    fig = px.bar(bin_df, x='Bin', y='WoE', text_auto='.2f', title=f"{sel_col}의 WoE 변화")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 사이드바에서 가설을 입력하고 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
