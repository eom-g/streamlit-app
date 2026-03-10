import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from optbinning import OptimalBinning
import streamlit.components.v1 as components
import sweetviz as sv

# --- 1. 페이지 및 AI 설정 ---
st.set_page_config(page_title="통신 세그먼트 Deep-Dive 분석기", layout="wide")

def get_gemini_insight(prompt, data_summary, user_api_key):
    api_key = user_api_key if user_api_key else st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "⚠️ API Key가 필요합니다."
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model = genai.GenerativeModel(available_models[0] if available_models else 'gemini-pro')
        full_prompt = f"당신은 전략 컨설턴트입니다. 아래 데이터 분석 결과를 바탕으로 사업팀이 즉시 실행할 'Targeting 전략'을 3줄로 제언하세요.\n\n분석 결과:\n{data_summary}\n질문: {prompt}"
        return model.generate_content(full_prompt).text
    except Exception as e: return f"❌ AI 분석 실패: {e}"

# --- 2. 드라마틱한 가상 데이터 생성 (차이를 극대화) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1200
    data = {
        '고객ID': range(10001, 10001 + n),
        '약정유형': np.random.choice(['SIM-only', '단말약정'], n),
        '요금제레벨': np.random.choice(['고가', '중가', '저가'], n),
        '단말유형': np.random.choice(['아이폰', '갤럭시 프리미엄', '갤럭시 중저가', '키즈폰'], n),
        '나이': np.random.randint(18, 75, n),
        '월평균매출_ARPU': np.random.uniform(30000, 100000, n),
        'OTT_접속건수': np.random.randint(0, 200, n),
        '쇼핑_접속건수': np.random.randint(0, 150, n),
        'SNS_접속건수': np.random.randint(0, 400, n),
        '금융_접속건수': np.random.randint(0, 100, n),
        '데이터사용량_GB': np.random.uniform(5, 100, n)
    }
    df = pd.DataFrame(data)
    
    # 세그먼트별 강력한 편향(Bias) 부여 - 분석 결과가 잘 나오도록 조정
    df.loc[df['약정유형'] == 'SIM-only', 'OTT_접속건수'] *= 5
    df.loc[df['약정유형'] == 'SIM-only', 'SNS_접속건수'] *= 3
    df.loc[df['단말유형'] == '아이폰', 'SNS_접속건수'] *= 4
    df.loc[df['요금제레벨'] == '고가', '금융_접속건수'] *= 6
    df.loc[df['요금제레벨'] == '고가', '월평균매출_ARPU'] += 50000
    df.loc[df['단말유형'] == '아이폰', '나이'] -= 20
    df['나이'] = df['나이'].clip(18, 80)
    return df

# --- 3. 메인 UI ---
def main():
    st.title("🚀 통신 세그먼트 정밀 대조 & 전략 도출")
    df = load_data()
    
    st.sidebar.header("📋 분석 설정")
    user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    dim = st.selectbox("비교 기준(Dimension):", ["약정유형", "요금제레벨", "단말유형"])
    vals = df[dim].unique().tolist()
    col1, col2 = st.columns(2)
    with col1: g_a = st.selectbox("그룹 A", vals, index=0)
    with col2: g_b = st.selectbox("그룹 B", vals, index=1)

    if st.button("Deep-Dive 분석 시작", use_container_width=True):
        df_a = df[df[dim] == g_a].copy()
        df_b = df[df[dim] == g_b].copy()
        
        # --- [1] IV 기반 Top 5 변수 선정 & OptBinning 상세 결과 ---
        st.header(f"📊 {g_a} vs {g_b} 핵심 차별점 랭킹")
        
        temp_a = df_a.copy(); temp_a['target'] = 0
        temp_b = df_b.copy(); temp_b['target'] = 1
        combined = pd.concat([temp_a, temp_b])
        
        features = [c for c in df.columns if c not in ['고객ID', dim, 'target']]
        iv_list = []
        binning_tables = {}

        for col in features:
            try:
                dtype = "numerical" if combined[col].dtype != 'object' else "categorical"
                optb = OptimalBinning(name=col, dtype=dtype, solver="cp")
                optb.fit(combined[col].values, combined['target'].values)
                bt = optb.binning_table.build()
                iv = bt.loc["전체", "IV"]
                iv_list.append({'feature': col, 'iv': iv})
                binning_tables[col] = bt
            except: continue
        
        top_5 = pd.DataFrame(iv_list).sort_values(by='iv', ascending=False).head(5)
        
        # 상위 5개 IV 메트릭 표시
        m_cols = st.columns(5)
        for i, row in enumerate(top_5.itertuples()):
            m_cols[i].metric(f"{i+1}위: {row.feature}", f"{row.iv:.3f}", "IV Score")

        # --- [2] 상위 변수별 하프 분포 차트 (Half-Graph) ---
        st.divider()
        st.header("📈 상위 변수 특성 비교 (Half-Density)")
        
        for feature in top_5['feature']:
            st.subheader(f"📍 {feature} 변수 정밀 분석")
            t1, t2 = st.columns([2, 1])
            
            with t1:
                # 하프 모양의 분포 그래프 (Density Plot)
                fig = go.Figure()
                fig.add_trace(go.Violin(x=df_a[feature], name=g_a, side='negative', line_color='blue'))
                fig.add_trace(go.Violin(x=df_b[feature], name=g_b, side='positive', line_color='orange'))
                fig.update_layout(violingap=0, violinmode='overlay', height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                # OptBinning 결과 테이블 요약
                st.caption(f"{feature}의 구간별 비중 및 WoE")
                st.dataframe(binning_tables[feature][['Bin', 'Count', 'Event', 'WoE', 'IV']].head(5), use_container_width=True)

        # --- [3] AI 인사이트 요약 ---
        st.divider()
        st.header("📝 AI 전략 마케팅 제언")
        analysis_summary = f"상위 5개 변수: {list(top_5['feature'])}, 각 IV 점수: {list(top_5['iv'].round(3))}"
        insight = get_gemini_insight(f"{g_a}와 {g_b} 세그먼트 차이 분석", analysis_summary, user_api_key)
        st.info(insight)

        # --- [4] Sweetviz 리포트 (전체 탐색용) ---
        st.divider()
        st.header("🔍 상세 데이터 탐색 (Sweetviz Full Report)")
        with st.spinner("상세 리포트 생성 중..."):
            report = sv.compare([df_a, g_a], [df_b, g_b])
            report.show_html("compare.html", open_browser=False)
            with open("compare.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
