import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components

# 1. 의존성 가드
try:
    from ydata_profiling import ProfileReport
except:
    ProfileReport = None

st.set_page_config(page_title="가설 기반 통신 데이터 분석", layout="wide")

# --- 2. 데이터 생성 로직 (동일) ---
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

# --- 3. 가설 키워드 매핑 함수 ---
def get_related_columns(hypothesis):
    # 가설에 포함된 키워드에 따라 관련 컬럼 반환
    mapping = {
        '쇼핑': ['접속_온라인쇼핑'],
        '음악': ['접속_음악스트리밍'],
        '영상': ['접속_OTT영상', '접속_SNS'],
        'OTT': ['접속_OTT영상'],
        '금융': ['접속_금융재테크'],
        '데이터': ['월_데이터사용량_GB'],
        '결합': ['결합유형', '가족결합_혜택이용여부'],
        '단말': ['단말사용기간_개월'],
        '이탈': ['이탈여부']
    }
    related = ['고객ID'] # 기본값
    for key, cols in mapping.items():
        if key in hypothesis:
            related.extend(cols)
    return list(set(related))

def main():
    st.title("🧪 통신 고객 가설 검증 샌드박스")
    df = load_telco_data()

    # 사이드바에서 가설 입력
    st.sidebar.header("1. 가설 설정")
    user_hypothesis = st.sidebar.text_area(
        "검증하고 싶은 가설을 입력하세요:",
        placeholder="예: OTT 사용량이 많은 고객은 결합 상품 이용률이 높을 것이다."
    )
    
    analyze_button = st.sidebar.button("가설 검증 시작")

    if analyze_button and user_hypothesis:
        # 가설 관련 변수 추출
        relevant_cols = get_related_columns(user_hypothesis)
        
        # 만약 매칭되는 키워드가 없으면 전체 보여주기 방지용
        if len(relevant_cols) <= 1:
             relevant_cols = df.columns.tolist()
             st.warning("⚠️ 명확한 분석 키워드를 찾지 못해 전체 데이터를 로드합니다.")

        filtered_df = df[relevant_cols]

        st.success(f"✔️ 입력하신 가설: {user_hypothesis}")
        st.info(f"🔍 가설과 관련된 주요 변수: {', '.join([c for c in relevant_cols if c != '고객ID'])}")

        tab1, tab2, tab3 = st.tabs(["📊 관련 데이터 추출", "📈 변수 상관관계", "💡 AI 가설 평가"])

        with tab1:
            st.subheader("가설 관련 데이터 뷰")
            st.dataframe(filtered_df.head(20))
            st.write(f"추출된 변수 개수: {len(relevant_cols)-1}개")

        with tab2:
            st.subheader("주요 변수 시각화")
            if len(relevant_cols) > 2:
                # 첫 번째 수치형 변수와 이탈여부 또는 나이의 관계 시각화
                num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                num_cols = [c for c in num_cols if c != '고객ID']
                
                if len(num_cols) >= 2:
                    fig = px.scatter(filtered_df, x=num_cols[0], y=num_cols[1], 
                                     color='결합유형' if '결합유형' in filtered_df.columns else None,
                                     title=f"{num_cols[0]}와 {num_cols[1]}의 관계")
                    st.plotly_chart(fig)
                else:
                    st.write("시각화를 위해 더 많은 키워드를 입력해주세요.")

        with tab3:
            st.subheader("Gemini AI 가설 검증 리포트")
            api_key = st.text_input("Gemini API Key", type="password")
            if st.button("AI 검증 리포트 생성"):
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    
                    data_summary = filtered_df.describe().to_string()
                    prompt = f"""
                    입력된 가설: {user_hypothesis}
                    관련 데이터 요약:
                    {data_summary}
                    
                    위 데이터를 바탕으로 이 가설이 타당한지 분석하고, 통신사 입장에서 어떤 마케팅 액션을 취해야 할지 한국어로 요약해줘.
                    """
                    with st.spinner("AI가 데이터를 분석 중입니다..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                else:
                    st.warning("API 키를 입력해주세요.")
    
    elif not analyze_button:
        st.info("👈 왼쪽 사이드바에 분석하고 싶은 가설을 입력하고 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
