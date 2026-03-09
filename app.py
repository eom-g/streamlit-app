import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning

# 페이지 설정
st.set_page_config(page_title="통신 가설 검증 대시보드", layout="wide")

@st.cache_data
def load_telco_data():
    np.random.seed(42)
    n_rows = 300
    data = {
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

def get_related_columns(hypothesis):
    mapping = {
        '쇼핑': ['접속_온라인쇼핑'],
        '음악': ['접속_음악스트리밍'],
        '영상': ['접속_OTT영상', '접속_SNS'],
        'OTT': ['접속_OTT영상'],
        '금융': ['접속_금융재테크'],
        '데이터': ['월_데이터사용량_GB'],
        '결합': ['결합유형', '가족결합_혜택이용여부'],
        '단말': ['단말사용기간_개월'],
        '나이': ['나이']
    }
    related = []
    for key, cols in mapping.items():
        if key in hypothesis:
            related.extend(cols)
    return list(set(related))

def main():
    st.title("🧪 통신 고객 가설 검증 샌드박스")
    df = load_telco_data()

    # 1. 사이드바 가설 입력
    with st.sidebar:
        st.header("1. 가설 설정")
        user_hypothesis = st.text_area("검증 가설 입력:", placeholder="예: OTT와 데이터 사용량은 관계가 있다.")
        analyze_clicked = st.button("가설 검증 시작")

    # 세션 상태 저장 (버튼 클릭 후에도 유지되도록)
    if analyze_clicked:
        st.session_state['run_analysis'] = True
        st.session_state['hypothesis'] = user_hypothesis

    if st.session_state.get('run_analysis'):
        hypothesis = st.session_state['hypothesis']
        relevant_cols = get_related_columns(hypothesis)
        
        # 타겟(이탈여부)은 분석을 위해 기본 포함
        if '이탈여부' not in relevant_cols:
            relevant_cols.append('이탈여부')

        # 분석할 데이터 필터링
        display_df = df[relevant_cols] if len(relevant_cols) > 1 else df

        st.success(f"✔️ 분석 중인 가설: {hypothesis}")
        
        tab1, tab2, tab3 = st.tabs(["📊 추출 데이터", "📈 변수 시각화", "💡 AI 분석"])

        with tab1:
            st.dataframe(display_df.head(10))

        with tab2:
            st.subheader("가설 기반 데이터 시각화")
            # 수치형 변수만 추출
            num_df = display_df.select_dtypes(include=[np.number])
            
            if num_df.shape[1] >= 2:
                # x축과 y축 변수 선택 (가설 관련 변수가 있으면 우선 선택)
                x_axis = num_df.columns[0]
                y_axis = num_df.columns[1] if num_df.shape[1] > 1 else num_df.columns[0]
                
                # 시각화 실행
                fig = px.scatter(
                    display_df, x=x_axis, y=y_axis, 
                    color='결합유형' if '결합유형' in display_df.columns else None,
                    trendline="ols", # 경향선 추가
                    title=f"[{x_axis}]와 [{y_axis}]의 상관관계 분석"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 추가 설명
                st.write(f"ℹ️ 가설 키워드에 따라 **{x_axis}** 변수와 **{y_axis}** 변수를 매칭하여 시각화했습니다.")
            else:
                st.warning("시각화를 위한 수치형 변수가 부족합니다. '데이터', '사용량', '접속' 등의 단어를 포함해 보세요.")

        with tab3:
            api_key = st.text_input("Gemini API Key", type="password")
            if st.button("AI 가설 평가 시작"):
                if api_key:
                    # AI 로직 실행 (생략 가능)
                    st.info("AI가 분석을 시작합니다...")
                else:
                    st.error("API 키를 입력해주세요.")

    else:
        st.info("👈 왼쪽 사이드바에 가설을 입력하고 [가설 검증 시작] 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
