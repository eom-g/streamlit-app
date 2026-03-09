import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components

# 1. 의존성 에러 방지용 가드
try:
    from ydata_profiling import ProfileReport
except (ImportError, AttributeError):
    ProfileReport = None

# 페이지 설정
st.set_page_config(page_title="통신 고객 라이프스타일 상세 분석", layout="wide")

# --- 2. 확장된 통신 샘플 데이터 생성 함수 ---
@st.cache_data
def load_telco_data():
    np.random.seed(42)
    n_rows = 200
    
    # 카테고리 정의
    membership_types = ['Bronze', 'Silver', 'Gold', 'VIP']
    combined_options = ['미결합', '유무선결합', '무무선결합(가족)']
    
    # 기본 정보 생성
    data = {
        '고객ID': range(1001, 1001 + n_rows),
        '나이': np.random.randint(18, 75, n_rows),
        '성별': np.random.choice(['남', '여'], n_rows),
        
        # 1. 웹앱 카테고리별 접속 건수 (한 달 기준)
        '접속_온라인쇼핑': np.random.randint(5, 100, n_rows),
        '접속_음악스트리밍': np.random.randint(10, 200, n_rows),
        '접속_OTT영상': np.random.randint(5, 150, n_rows),
        '접속_SNS': np.random.randint(20, 300, n_rows),
        '접속_금융재테크': np.random.randint(2, 80, n_rows),
        
        # 2. 사용량 및 단말 정보
        '월_데이터사용량_GB': np.random.uniform(2, 120, n_rows),
        '월_평균사용시간_시간': np.random.uniform(20, 300, n_rows),
        '단말사용기간_개월': np.random.randint(1, 60, n_rows),
        
        # 3. 멤버십 및 혜택
        '멤버십등급': np.random.choice(membership_types, n_rows),
        '멤버십_혜택사용건수': np.random.randint(0, 20, n_rows),
        '멤버십_실사용금액': np.random.randint(0, 100000, n_rows),
        
        # 4. 결합 정보
        '결합유형': np.random.choice(combined_options, n_rows),
        '가족결합_혜택이용여부': np.random.choice(['Y', 'N'], n_rows, p=[0.45, 0.55]),
        
        # 타겟 변수 (이탈 예측용 샘플)
        '이탈여부': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # 주 사용 카테고리 계산 (접속 건수가 가장 많은 곳)
    visit_cols = ['접속_온라인쇼핑', '접속_음악스트리밍', '접속_OTT영상', '접속_SNS', '접속_금융재테크']
    df['주사용_카테고리'] = df[visit_cols].idxmax(axis=1).str.replace('접속_', '')
    
    return df

def main():
    st.title("📱 통신 고객 라이프스타일 및 결합 혜택 정밀 분석")
    
    # 데이터 로드
    df = load_telco_data()
    
    st.sidebar.success("상세 샘플 데이터 로드 완료")
    st.sidebar.info(f"데이터 총량: {df.shape[0]}명 고객")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 현황", "🔍 카테고리별 EDA", "📈 구간 최적화", "🤖 AI 마케팅"])

    # --- Tab 1: 데이터 미리보기 ---
    with tab1:
        st.subheader("고객별 상세 라이프스타일 데이터")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**카테고리별 평균 접속 건수**")
            visit_cols = ['접속_온라인쇼핑', '접속_음악스트리밍', '접속_OTT영상', '접속_SNS', '접속_금융재테크']
            st.bar_chart(df[visit_cols].mean())
        with col2:
            st.write("**결합 유형별 분포**")
            st.write(df['결합유형'].value_counts())

    # --- Tab 2: YData Profiling ---
    with tab2:
        st.subheader("데이터 상세 프로파일링 리포트")
        if st.button("EDA 리포트 생성"):
            if ProfileReport is not None:
                with st.spinner("다차원 리포트를 생성 중입니다..."):
                    pr = ProfileReport(df, explorative=True, minimal=True)
                    components.html(pr.to_html(), height=800, scrolling=True)
            else:
                st.error("리포트 생성 라이브러리가 로드되지 않았습니다.")

    # --- Tab 3: OptBinning (금융재테크 접속 vs 이탈) ---
    with tab3:
        st.subheader("구간 최적화: 금융/재테크 접속 건수 분석")
        st.write("금융 앱 접속 빈도와 고객 유지율 간의 상관관계를 최적의 구간으로 나눕니다.")
        
        x = df['접속_금융재테크'].values
        y = df['이탈여부'].values
        
        optb = OptimalBinning(name="finance_visits", dtype="numerical", solver="cp")
        optb.fit(x, y)
        
        st.write("**Optimal Binning Table**")
        st.dataframe(optb.binning_table.build())
        
        # WOE 그래프
        fig = px.bar(optb.binning_table.build()[:-1], x='Bin', y='WoE', title="금융 앱 접속 구간별 WoE 지표")
        st.plotly_chart(fig)

    # --- Tab 4: Gemini AI 마케팅 전략 ---
    with tab4:
        st.subheader("AI 맞춤형 고객 인사이트")
        api_key = st.text_input("Gemini API Key", type="password")
        
        if st.button("분석 결과 기반 마케팅 제안"):
            if not api_key:
                st.warning("API 키를 입력하세요.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    # AI에게 줄 데이터 요약
                    top_cat_dist = df['주사용_카테고리'].value_counts().to_dict()
                    combined_dist = df['결합유형'].value_counts().to_dict()
                    
                    prompt = f"""
                    다음 통신사 고객 데이터를 분석해줘:
                    1. 주사용 서비스 분포: {top_cat_dist}
                    2. 결합 상태: {combined_dist}
                    3. 가족결합 혜택 사용 여부: {df['가족결합_혜택이용여부'].value_counts().to_dict()}
                    
                    위 데이터를 바탕으로 'OTT 주사용자'를 '유무선 결합 상품'으로 유도하기 위한 구체적인 마케팅 문구와 전략을 한국어로 3가지 작성해줘.
                    """
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"에러 발생: {e}")

if __name__ == "__main__":
    main()
