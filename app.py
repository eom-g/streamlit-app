import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import sweetviz as sv
from ydata_profiling import ProfileReport

# --- 1. 페이지 설정 및 Gemini 설정 ---
st.set_page_config(page_title="통신 가설 검증 AI 샌드박스", layout="wide")

def get_gemini_insight(prompt, data_summary, api_key):
    if not api_key:
        return "⚠️ Gemini API Key를 입력하면 AI 인사이트를 볼 수 있습니다."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        full_prompt = f"""
        당신은 통신사 전문 데이터 분석가입니다. 
        사용자의 가설: {prompt}
        데이터 요약 통계: {data_summary}
        
        위 데이터를 바탕으로 가설을 검증하고, 사업팀이 즉시 실행 가능한 마케팅 인사이트를 3줄로 요약하세요.
        말투는 정중하고 전문적인 비즈니스 문체를 사용하세요.
        """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 요약 중 오류 발생: {e}"

# --- 2. 데이터 로드 (기존 동일) ---
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
    df = pd.DataFrame(data)
    df.loc[df['약정유형'] == 'SIM-only', '데이터사용량_GB'] += 20
    return df

# --- 3. 메인 UI ---
def main():
    st.title("🧪 통신 고객 가설 검증 AI 샌드박스")
    df = load_data()

    # 사이드바: 모드 선택
    st.sidebar.header("🛠️ 분석 모드 선택")
    mode = st.sidebar.radio("모드를 선택하세요:", ["자연어 가설 분석", "세그먼트 1:1 비교"])
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

    if mode == "자연어 가설 분석":
        st.subheader("💡 가설 기반 자동 분석")
        user_hypothesis = st.text_area("검증하고 싶은 가설을 입력하세요:", 
                                     placeholder="예: 20대 아이폰 유저는 고가 요금제 사용 비중이 높고 이탈률이 낮을 것이다.")
        
        if st.button("가설 검증 시작"):
            # 가설 관련 키워드로 데이터 필터링 시뮬레이션 (실제로는 더 복잡한 로직 가능)
            st.info("🔍 LLM이 가설을 분석하여 관련 데이터를 추출 중입니다...")
            
            # (가설 검증을 위한 데이터 요약 생성)
            summary_stats = df.describe().to_string()
            
            with st.expander("📝 AI 인사이트 요약", expanded=True):
                insight = get_gemini_insight(user_hypothesis, summary_stats, api_key)
                st.markdown(insight)

            # 가설 관련 시각화 (예시: 나이대별 요금제 분포)
            fig = px.histogram(df, x="나 age" if "나이" in user_hypothesis else "데이터사용량_GB", 
                               color="요금제레벨", barmode="group", title="가설 관련 데이터 분포")
            st.plotly_chart(fig, use_container_width=True)

    else:
        # 기존 세그먼트 비교 로직 (Sweetviz 활용)
        st.subheader("👥 세그먼트 1:1 대조 분석")
        dimension = st.sidebar.selectbox("비교 디멘젼:", ["약정유형", "요금제레벨", "단말유형"])
        vals = df[dimension].unique().tolist()
        g_a = st.sidebar.selectbox("그룹 A", vals, index=0)
        g_b = st.sidebar.selectbox("그룹 B", vals, index=1)

        if st.button("비교 리포트 및 AI 요약 생성"):
            report = sv.compare([df[df[dimension]==g_a], g_a], [df[df[dimension]==g_b], g_b], target_feat='이탈여부')
            report.show_html('compare.html', open_browser=False)
            
            # AI 요약 (두 그룹의 평균 차이 전달)
            diff_summary = f"그룹A({g_a}) 평균 ARPU: {df[df[dimension]==g_a]['월평균매출_ARPU'].mean():.0f}, 그룹B({g_b}) 평균 ARPU: {df[df[dimension]==g_b]['월평균매출_ARPU'].mean():.0f}"
            
            with st.expander("📝 AI 세그먼트 비교 요약", expanded=True):
                insight = get_gemini_insight(f"{g_a}와 {g_b} 그룹 비교", diff_summary, api_key)
                st.markdown(insight)
            
            with open('compare.html', 'r', encoding='utf-8') as f:
                components.html(f.read(), height=1000, scrolling=True)

if __name__ == "__main__":
    main()
