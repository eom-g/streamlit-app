import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import sweetviz as sv
from ydata_profiling import ProfileReport

# --- 1. 페이지 및 AI 설정 ---
st.set_page_config(page_title="통신 라이프스타일 분석 샌드박스", layout="wide")

def get_gemini_insight(prompt, data_summary, user_api_key):
    api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ API Key가 필요합니다. 사이드바에 입력해주세요."
    
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() 
                            if 'generateContent' in m.supported_generation_methods]
        if not available_models: return "❌ 가용 모델이 없습니다."
        
        model = genai.GenerativeModel(available_models[0])
        full_prompt = f"통신 마케팅 전문가로서 분석하세요.\n데이터: {data_summary}\n주제: {prompt}\n결론/제언 3줄 요약."
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 연결 실패: {str(e)}"

# --- 2. IV 계산 함수 (에러 방어 로직 강화) ---
def calculate_top_iv_features(df_a, df_b, features):
    iv_results = []
    if df_a.empty or df_b.empty:
        return pd.DataFrame(columns=['feature', 'iv'])
        
    temp_a = df_a.copy(); temp_a['target_group'] = 0
    temp_b = df_b.copy(); temp_b['target_group'] = 1
    combined = pd.concat([temp_a, temp_b])
    
    for col in features:
        if col in ['고객ID', 'target_group', '이탈여부']: continue
        try:
            # 고유값이 1개뿐이면 IV 계산 의미 없음
            if combined[col].nunique() <= 1: continue
            
            dtype = "numerical" if combined[col].dtype != 'object' else "categorical"
            optb = OptimalBinning(name=col, dtype=dtype, solver="cp")
            optb.fit(combined[col].values, combined['target_group'].values)
            iv = optb.binning_table.build().loc["전체", "IV"]
            
            # IV가 유효한 숫자인 경우만 추가
            if pd.notnull(iv) and iv != np.inf:
                iv_results.append({'feature': col, 'iv': iv})
        except:
            continue
            
    if not iv_results:
        return pd.DataFrame(columns=['feature', 'iv'])
        
    return pd.DataFrame(iv_results).sort_values(by='iv', ascending=False).head(5)

# --- 3. 가상 데이터 생성 ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    data = {
        '고객ID': range(10001, 10001 + n),
        '약정유형': np.random.choice(['SIM-only', '단말약정'], n),
        '요금제레벨': np.random.choice(['고가', '중가', '저가'], n),
        '단말유형': np.random.choice(['아이폰', '갤럭시 프리미엄', '갤럭시 중저가', '기타'], n),
        '나이': np.random.randint(18, 75, n),
        '월평균매출_ARPU': np.random.uniform(30000, 100000, n),
        'OTT_접속일수': np.random.randint(0, 31, n),
        'OTT_접속건수': np.random.randint(0, 500, n),
        '쇼핑_접속일수': np.random.randint(0, 31, n),
        '쇼핑_접속건수': np.random.randint(0, 300, n),
        'SNS_접속일수': np.random.randint(0, 31, n),
        'SNS_접속건수': np.random.randint(0, 1000, n),
        '금융_접속일수': np.random.randint(0, 31, n),
        '금융_접속건수': np.random.randint(0, 150, n)
    }
    df = pd.DataFrame(data)
    # 그룹 간 차이를 유도하기 위한 로직
    df.loc[df['약정유형'] == 'SIM-only', 'OTT_접속일수'] += 10
    df.loc[df['단말유형'] == '아이폰', 'SNS_접속건수'] *= 1.8
    df['데이터사용량_GB'] = ((df['OTT_접속건수']*0.2) + (df['SNS_접속건수']*0.05) + 10).round(1)
    return df

# --- 4. 메인 UI ---
def main():
    st.title("📊 통신 고객 행태 분석 및 IV 랭킹")
    df = load_data()

    st.sidebar.header("⚙️ 분석 설정")
    mode = st.sidebar.radio("모드:", ["세그먼트 1:1 비교", "자연어 가설 분석"])
    user_api_key = st.sidebar.text_input("Gemini API Key", type="password")

    if mode == "세그먼트 1:1 비교":
        dim = st.selectbox("비교 기준:", ["약정유형", "요금제레벨", "단말유형"])
        vals = df[dim].unique().tolist()
        c1, c2 = st.columns(2)
        with c1: g_a = st.selectbox("그룹 A:", vals, index=0)
        with c2: g_b = st.selectbox("그룹 B:", vals, index=1)

        if st.button("분석 실행"):
            df_a = df[df[dim] == g_a]
            df_b = df[df[dim] == g_b]

            # IV 랭킹 계산 및 예외 처리
            analysis_cols = [c for c in df.columns if c not in ['고객ID']]
            top_iv = calculate_top_iv_features(df_a, df_b, analysis_cols)
            
            st.markdown(f"### 🏆 {g_a} vs {g_b} 핵심 차별 변수")
            if not top_iv.empty:
                iv_cols = st.columns(len(top_iv))
                for idx, row in enumerate(top_iv.itertuples()):
                    with iv_cols[idx]:
                        st.metric(label=f"{idx+1}위: {row.feature}", value=f"{row.iv:.3f}")
            else:
                st.warning("🧐 두 그룹 간에 통계적으로 유의미한 차이를 보이는 변수가 없습니다. 다른 그룹을 선택해보세요.")

            # AI 인사이트 및 Sweetviz
            if not top_iv.empty:
                with st.expander("📝 AI 전략 분석", expanded=True):
                    summary = f"그룹A({g_a}), 그룹B({g_b}). 주요 차이: {list(top_iv['feature'])}"
                    st.info(get_gemini_insight("두 고객군 특성 비교", summary, user_api_key))

            with st.spinner("상세 리포트 생성 중..."):
                report = sv.compare([df_a, g_a], [df_b, g_b])
                report.show_html("compare.html", open_browser=False)
                with open("compare.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=1000, scrolling=True)

    elif mode == "자연어 가설 분석":
        hypothesis = st.text_area("가설 입력:", "아이폰 유저는 SNS 접속량이 많을 것이다.")
        if st.button("검증"):
            st.success(get_gemini_insight(hypothesis, "데이터 요약 생략", user_api_key))
            st.plotly_chart(px.box(df, x="단말유형", y="SNS_접속건수"))

if __name__ == "__main__":
    main()
