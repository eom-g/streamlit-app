import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import sweetviz as sv
from ydata_profiling import ProfileReport
import os

# --- 1. 페이지 설정 및 AI 엔진 설정 ---
st.set_page_config(page_title="통신 고객 가설 검증 AI 샌드박스", layout="wide")

def get_gemini_insight(prompt, data_summary, user_api_key):
    # 사이드바 입력 키 우선, 없으면 Secrets 확인
    api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        return "⚠️ API Key가 필요합니다. 사이드바에 입력하거나 Streamlit Secrets에 등록해주세요."
    
    try:
        genai.configure(api_key=api_key)
        # 안정적인 1.5-flash 모델 사용 (속도 및 비용 최적화)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        full_prompt = f"""
        당신은 통신사 데이터 기반 마케팅 전략가입니다.
        
        [분석 데이터 요약]
        {data_summary}
        
        [사용자 분석 가설]
        {prompt}
        
        위 데이터를 바탕으로 가설의 타당성을 평가하고, 사업팀이 즉시 실행할 수 있는 전략적 제언(Action Item)을 3가지 포인트로 요약하세요. 
        데이터에 근거하여 구체적인 수치를 언급하면 더 좋습니다.
        """
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        if "API_KEY_INVALID" in str(e):
            return "❌ API Key가 유효하지 않습니다. 다시 확인해주세요."
        return f"❌ AI 요약 중 오류 발생: {str(e)}"

# --- 2. 가상 데이터 생성 (사업팀 디멘젼 반영) ---
@st.cache_data
def load_telco_data():
    np.random.seed(42)
    n_rows = 1000
    data = {
        '고객ID': range(10001, 10001 + n_rows),
        '약정유형': np.random.choice(['SIM-only', '단말약정'], n_rows, p=[0.4, 0.6]),
        '요금제레벨': np.random.choice(['고가(8.5만↑)', '중가(5.5만↑)', '저가(5.5만↓)'], n_rows),
        '단말유형': np.random.choice(['아이폰 프리미엄', '갤럭시 프리미엄', '갤럭시 중저가', '키즈폰/기타'], n_rows),
        '나이': np.random.randint(18, 75, n_rows),
        '데이터사용량_GB': np.random.uniform(2, 200, n_rows),
        '월평균매출_ARPU': np.random.uniform(30000, 110000, n_rows),
        '이탈여부': np.random.choice([0, 1], n_rows, p=[0.82, 0.18])
    }
    df = pd.DataFrame(data)
    # 가설 검증용 로직: SIM-only이면서 아이폰이면 데이터 사용량이 높도록 조정
    df.loc[(df['약정유형']=='SIM-only') & (df['단말유형']=='아이폰 프리미엄'), '데이터사용량_GB'] *= 1.5
    return df

# --- 3. 메인 UI 구성 ---
def main():
    st.title("🧪 통신 고객 가설 검증 & EDA 자동화")
    st.markdown("사업팀의 가설을 AI가 검증하고, 분석가용 상세 리포트를 생성합니다.")
    
    df = load_telco_data()

    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 설정")
    mode = st.sidebar.radio("분석 모드:", ["자연어 가설 검증", "세그먼트 1:1 비교", "전체 데이터 프로파일링"])
    user_api_key = st.sidebar.text_input("Gemini API Key (입력 후 Enter)", type="password")
    
    st.sidebar.divider()
    st.sidebar.info("💡 **Tip:** Streamlit Secrets에 GEMINI_API_KEY를 등록하면 편리합니다.")

    # 모드 1: 자연어 가설 검증 (사업팀용)
    if mode == "자연어 가설 검증":
        st.subheader("💡 비즈니스 가설 분석")
        col1, col2 = st.columns([1.2, 0.8])
        
        with col1:
            hypothesis = st.text_area("검증할 가설을 입력하세요:", 
                                     value="아이폰 프리미엄 단말을 사용하는 SIM-only 고객은 데이터 사용량이 많고 매출 기여도가 높을 것이다.")
        
        if st.button("AI 가설 검증 실행", use_container_width=True):
            # 분석용 요약 통계량 생성
            summary_df = df.groupby(['약정유형', '단말유형']).agg({
                '데이터사용량_GB': 'mean',
                '월평균매출_ARPU': 'mean',
                '이탈여부': 'mean'
            }).round(2).to_string()
            
            with st.spinner("AI 분석가 가동 중..."):
                # 1. AI 인사이트 요약
                insight = get_gemini_insight(hypothesis, summary_df, user_api_key)
                st.markdown("### 📝 AI 인사이트 요약")
                st.success(insight)
                
                # 2. 가설 기반 시각화
                st.markdown("### 📊 데이터 근거 (Visualization)")
                fig = px.scatter(df, x="데이터사용량_GB", y="월평균매출_ARPU", 
                                 color="약정유형", symbol="단말유형",
                                 title="약정 및 단말유형별 데이터-매출 상관관계")
                st.plotly_chart(fig, use_container_width=True)

    # 모드 2: 세그먼트 1:1 비교 (사업팀/분석가용)
    elif mode == "세그먼트 1:1 비교":
        st.subheader("👥 세그먼트 대조 분석 (Sweetviz)")
        
        dim = st.selectbox("비교할 기준(Dimension):", ["약정유형", "요금제레벨", "단말유형"])
        vals = df[dim].unique().tolist()
        
        c1, c2 = st.columns(2)
        with c1: g_a = st.selectbox("그룹 A 선택:", vals, index=0)
        with c2: g_b = st.selectbox("그룹 B 선택:", vals, index=1)

        if st.button("세그먼트 비교 리포트 생성"):
            df_a = df[df[dim] == g_a]
            df_b = df[df[dim] == g_b]
            
            with st.spinner("리포트 생성 중... (약 10초 소요)"):
                report = sv.compare([df_a, g_a], [df_b, g_b], target_feat='이탈여부')
                report.show_html("compare.html", open_browser=False)
                
                # AI 요약 병행
                summary_text = f"그룹A({g_a}) 이탈률: {df_a['이탈여부'].mean():.2%}, 그룹B({g_b}) 이탈률: {df_b['이탈여부'].mean():.2%}"
                insight = get_gemini_insight(f"{g_a}와 {g_b} 세그먼트 차이 분석", summary_text, user_api_key)
                st.info(f"✨ AI 요약: {insight}")
                
                with open("compare.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=1000, scrolling=True)

    # 모드 3: 전체 데이터 프로파일링 (분석가 전용)
    elif mode == "전체 데이터 프로파일링":
        st.subheader("📋 데이터셋 전체 건강검진 (YData)")
        if st.button("상세 프로파일링 시작"):
            with st.spinner("데이터 분석 중..."):
                profile = ProfileReport(df, explorative=True, minimal=True)
                components.html(profile.to_html(), height=1000, scrolling=True)

if __name__ == "__main__":
    main()
