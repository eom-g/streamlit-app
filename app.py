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
        # 가용 모델 자동 조회 및 선택 (404 에러 방지)
        available_models = [m.name for m in genai.list_models() 
                            if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            return "❌ 가용 모델이 없습니다."
        
        model = genai.GenerativeModel(available_models[0])
        full_prompt = f"통신 마케팅 전문가로서 다음 데이터를 분석하세요.\n데이터 요약: {data_summary}\n분석 주제: {prompt}\n\n결론과 마케팅 제언을 3줄 요약하세요."
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 엔진 연결 실패: {str(e)}"

# --- 2. IV(Information Value) 계산 함수 ---
def calculate_top_iv_features(df_a, df_b, features):
    iv_results = []
    temp_a = df_a.copy(); temp_a['target_group'] = 0
    temp_b = df_b.copy(); temp_b['target_group'] = 1
    combined = pd.concat([temp_a, temp_b])
    
    for col in features:
        if col in ['고객ID', 'target_group']: continue
        try:
            dtype = "numerical" if combined[col].dtype != 'object' else "categorical"
            optb = OptimalBinning(name=col, dtype=dtype, solver="cp")
            optb.fit(combined[col].values, combined['target_group'].values)
            iv = optb.binning_table.build().loc["전체", "IV"]
            iv_results.append({'feature': col, 'iv': iv})
        except:
            continue
    return pd.DataFrame(iv_results).sort_values(by='iv', ascending=False).head(5)

# --- 3. 가상 데이터 생성 (웹/앱 접속 로그 반영) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    data = {
        '고객ID': range(10001, 10001 + n),
        '약정유형': np.random.choice(['SIM-only', '단말약정'], n),
        '요금제레벨': np.random.choice(['고가', '중가', '저가'], n),
        '단말유형': np.random.choice(['아이폰', '갤럭시 프리미엄', '갤럭시 중저가', '키즈폰/기타'], n),
        '나이': np.random.randint(18, 75, n),
        '월평균매출_ARPU': np.random.uniform(30000, 100000, n),
        
        # 웹/앱 접속 로그 (일수 및 건수)
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

    # 상관관계 주입: SIM-only & 아이폰 유저는 SNS와 OTT 사용량이 높음
    df.loc[df['약정유형'] == 'SIM-only', 'OTT_접속일수'] += 7
    df.loc[df['단말유형'] == '아이폰', 'SNS_접속건수'] *= 1.5
    
    # 데이터사용량은 접속건수의 총합으로 시뮬레이션
    df['데이터사용량_GB'] = ((df['OTT_접속건수']*0.2) + (df['SNS_접속건수']*0.05) + np.random.uniform(5, 15, n)).round(1)
    
    return df

# --- 4. 메인 UI ---
def main():
    st.title("📊 통신 고객 라이프스타일 EDA 자동화")
    df = load_data()

    st.sidebar.header("⚙️ 분석 설정")
    mode = st.sidebar.radio("모드 선택:", ["세그먼트 1:1 비교", "자연어 가설 분석", "데이터 상세 탐색"])
    user_api_key = st.sidebar.text_input("Gemini API Key", type="password")

    if mode == "세그먼트 1:1 비교":
        st.subheader("👥 세그먼트 간 행태 대조 분석")
        
        dim = st.selectbox("비교 기준 선택:", ["약정유형", "요금제레벨", "단말유형"])
        vals = df[dim].unique().tolist()
        c1, c2 = st.columns(2)
        with c1: g_a = st.selectbox("그룹 A:", vals, index=0)
        with c2: g_b = st.selectbox("그룹 B:", vals, index=1)

        if st.button("정밀 비교 리포트 생성"):
            df_a = df[df[dim] == g_a]
            df_b = df[df[dim] == g_b]

            # A. IV 랭킹 (무엇이 가장 다른가?)
            st.markdown(f"### 🏆 {g_a} vs {g_b} 차별점 Top 5 (IV 기준)")
            analysis_cols = [c for c in df.columns if c not in ['고객ID']]
            top_iv = calculate_top_iv_features(df_a, df_b, analysis_cols)
            
            iv_cols = st.columns(5)
            for idx, row in enumerate(top_iv.itertuples()):
                with iv_cols[idx]:
                    st.metric(label=f"{idx+1}위: {row.feature}", value=f"{row.iv:.3f}")

            # B. AI 인사이트
            with st.expander("📝 AI 전략 리포트", expanded=True):
                summary = f"그룹A({g_a}) n={len(df_a)}, 그룹B({g_b}) n={len(df_b)}. 주요 차이변수 IV 순위: {list(top_iv['feature'])}"
                insight = get_gemini_insight(f"{g_a}와 {g_b} 고객의 웹/앱 사용 패턴 및 마케팅 제언", summary, user_api_key)
                st.info(insight)

            # C. Sweetviz 비교 (하프 그래프 뷰)
            with st.spinner("상세 분포 그래프 생성 중..."):
                # target_feat를 빼서 '행태 비교' 모드로 실행
                report = sv.compare([df_a, g_a], [df_b, g_b])
                report.show_html("compare.html", open_browser=False)
                with open("compare.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=1000, scrolling=True)

    elif mode == "자연어 가설 분석":
        st.subheader("💡 가설 기반 자동 검증")
        hypothesis = st.text_area("사업팀 가설 입력:", "아이폰을 사용하는 고가 요금제 고객은 SNS와 쇼핑 접속 건수가 타 그룹보다 압도적으로 높을 것이다.")
        if st.button("가설 검증 실행"):
            summary = df.groupby('단말유형')[['SNS_접속건수', '쇼핑_접속건수']].mean().to_string()
            insight = get_gemini_insight(hypothesis, summary, user_api_key)
            st.success(insight)
            st.plotly_chart(px.box(df, x="단말유형", y="SNS_접속건수", color="요금제레벨", title="단말 및 요금제별 SNS 사용 행태"))

    elif mode == "데이터 상세 탐색":
        st.subheader("📋 데이터셋 전체 프로파일링")
        if st.button("YData 리포트 생성"):
            profile = ProfileReport(df, minimal=True)
            components.html(profile.to_html(), height=1000, scrolling=True)

if __name__ == "__main__":
    main()
