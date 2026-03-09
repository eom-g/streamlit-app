import sys

# 1. pkg_resources 에러 방지를 위한 런타임 패치
try:
    import pkg_resources
except ImportError:
    import pip
    pip.main(['install', 'setuptools'])
    import pkg_resources

# 2. 만약 위 방법으로도 안될 경우를 대비한 가상 경로 설정
if not hasattr(pkg_resources, 'resource_filename'):
    import setuptools.pkg_resources as pkg_resources

# --- 이제 기존 라이브러리를 불러옵니다 ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import sweetviz as sv  # 이제 에러 없이 로드될 것입니다.
import streamlit.components.v1 as components

# 1. 의존성 가드 (패키지 로드 확인)
try:
    from ydata_profiling import ProfileReport
except:
    ProfileReport = None

# 페이지 설정
st.set_page_config(page_title="Awesome EDA - Telco Lifestyle", layout="wide")

# --- 2. 데이터 생성 로직 (확장된 통신 데이터) ---
@st.cache_data
def load_telco_data():
    np.random.seed(42)
    n_rows = 300
    data = {
        '고객ID': range(1001, 1001 + n_rows),
        '나이': np.random.randint(18, 75, n_rows),
        '접속_온라인쇼핑': np.random.randint(5, 100, n_rows),
        '접속_음악스트리밍': np.random.randint(10, 200, n_rows),
        '접속_OTT영상': np.random.randint(5, 150, n_rows),
        '접속_SNS': np.random.randint(20, 300, n_rows),
        '접속_금융재테크': np.random.randint(2, 80, n_rows),
        '월_데이터사용량_GB': np.random.uniform(2, 120, n_rows),
        '단말사용기간_개월': np.random.randint(1, 60, n_rows),
        '결합유형': np.random.choice(['미결합', '유무선결합', '무무선결합(가족)'], n_rows),
        '가족결합_혜택이용여부': np.random.choice(['Y', 'N'], n_rows),
        '이탈여부': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# --- 키워드 매핑 (가설에 따른 변수 자동 선택) ---
def get_related_columns(hypothesis):
    mapping = {
        '쇼핑': ['접속_온라인쇼핑'], '음악': ['접속_음악스트리밍'], '영상': ['접속_OTT영상'],
        'OTT': ['접속_OTT영상'], 'SNS': ['접속_SNS'], '금융': ['접속_금융재테크'],
        '데이터': ['월_데이터사용량_GB'], '결합': ['결합유형', '가족결합_혜택이용여부'],
        '단말': ['단말사용기간_개월'], '나이': ['나이'], '이탈': ['이탈여부']
    }
    related = ['고객ID']
    for key, cols in mapping.items():
        if key in hypothesis: related.extend(cols)
    return list(set(related))

def main():
    st.title("🚀 Awesome EDA: 통신 가설 검증 플랫폼")
    df = load_telco_data()

    # 사이드바 설정
    with st.sidebar:
        st.header("1. 분석 설정")
        user_hypothesis = st.text_area("검증할 가설을 입력하세요:", value="OTT 영상 접속이 많은 고객은 데이터 사용량도 많을 것이다.")
        st.markdown("---")
        api_key = st.text_input("Gemini API Key", type="password")
        analyze_btn = st.button("가설 기반 분석 실행", use_container_width=True)

    if analyze_btn or st.session_state.get('active'):
        st.session_state['active'] = True
        
        # 가설 관련 컬럼 추출
        rel_cols = get_related_columns(user_hypothesis)
        if len(rel_cols) <= 1: rel_cols = df.columns.tolist()
        
        target_df = df[rel_cols]

        # 메인 탭 구성 (Sweetviz, YData, OptBinning 전면 배치)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Selected Data", 
            "✨ YData Profiling", 
            "🍭 Sweetviz", 
            "📈 OptBinning", 
            "🤖 AI Insight"
        ])

        with tab1:
            st.subheader("가설 관련 데이터 요약")
            st.dataframe(target_df.head(10))
            st.write(target_df.describe())

        with tab2:
            st.subheader("YData Profiling Report")
            if ProfileReport is not None:
                if st.button("Generate YData Report"):
                    with st.spinner("전문 통계 리포트 생성 중..."):
                        # 가설 관련 변수 위주로 리포트 생성
                        pr = ProfileReport(target_df, explorative=True, minimal=True)
                        components.html(pr.to_html(), height=900, scrolling=True)
            else:
                st.error("YData-Profiling 라이브러리를 로드할 수 없습니다.")

        with tab3:
            st.subheader("Sweetviz Comparison Report")
            if st.button("Generate Sweetviz Report"):
                with st.spinner("비교 분석 리포트 생성 중..."):
                    # 이탈여부를 기준으로 Sweetviz 분석
                    target_col = '이탈여부' if '이탈여부' in target_df.columns else None
                    report = sv.analyze(target_df, target_feat=target_col)
                    report.show_html(filepath='sweetviz_report.html', open_browser=False)
                    with open('sweetviz_report.html', 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=900, scrolling=True)

        with tab4:
            st.subheader("Optimal Binning (변수 최적화)")
            # 수치형 변수 하나 선택해서 이탈여부와의 관계 분석
            num_cols = target_df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c not in ['고객ID', '이탈여부']]
            
            if num_cols:
                sel_col = st.selectbox("분석할 변수 선택:", num_cols)
                optb = OptimalBinning(name=sel_col, dtype="numerical", solver="cp")
                optb.fit(target_df[sel_col].values, df['이탈여부'].values)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Binning Table**")
                    st.dataframe(optb.binning_table.build())
                with col2:
                    st.write("**WoE Chart**")
                    fig = px.bar(optb.binning_table.build()[:-1], x='Bin', y='WoE', color='WoE')
                    st.plotly_chart(fig)
            else:
                st.warning("분석할 수치형 변수가 부족합니다.")

        with tab5:
            st.subheader("Gemini AI 가설 검증")
            if api_key:
                if st.button("AI 인사이트 도출"):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"가설: {user_hypothesis}\n데이터 요약: {target_df.describe().to_string()}\n이 데이터를 기반으로 가설을 검증하고 마케팅 전략을 제안해줘."
                    response = model.generate_content(prompt)
                    st.write(response.text)
            else:
                st.info("사이드바에 API 키를 입력하면 AI 분석이 활성화됩니다.")

    else:
        st.info("👈 왼쪽 사이드바에서 가설을 입력하고 분석 버튼을 눌러주세요. Awesome EDA 라이브러리들이 작동을 시작합니다!")

if __name__ == "__main__":
    main()
