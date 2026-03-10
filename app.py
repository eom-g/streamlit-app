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
        full_prompt = f"전략 컨설턴트로서 아래 데이터를 분석하고 타겟팅 전략을 3줄 요약하세요.\n\n데이터: {data_summary}\n질문: {prompt}"
        return model.generate_content(full_prompt).text
    except Exception as e: return f"❌ AI 분석 실패: {e}"

# --- 2. [강력 수정] 무조건 차이가 나도록 설계된 가상 데이터 ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1500
    
    # 1. 기본 카테고리 데이터 생성
    약정유형_list = ['SIM-only', '단말약정']
    요금제_list = ['고가', '중가', '저가']
    단말_list = ['아이폰', '갤럭시 프리미엄', '갤럭시 중저가', '키즈폰']
    
    data = {
        '고객ID': range(10001, 10001 + n),
        '약정유형': np.random.choice(약정유형_list, n),
        '요금제레벨': np.random.choice(요금제_list, n),
        '단말유형': np.random.choice(단말_list, n),
        '나이': np.random.randint(20, 70, n),
        '월평균매출_ARPU': np.random.randint(30000, 80000, n),
        'OTT_접속건수': np.random.randint(5, 100, n),
        '쇼핑_접속건수': np.random.randint(5, 80, n),
        'SNS_접속건수': np.random.randint(10, 150, n),
        '금융_접속건수': np.random.randint(2, 40, n),
        '데이터사용량_GB': np.random.uniform(10, 50, n)
    }
    df = pd.DataFrame(data)
    
    # 2. [핵심] 모든 그룹 조합에 대해 극단적인 차이(Bias) 강제 주입
    # 약정유형별 차이
    df.loc[df['약정유형'] == 'SIM-only', 'OTT_접속건수'] *= 4.5
    df.loc[df['약정유형'] == 'SIM-only', '데이터사용량_GB'] *= 3.0
    
    # 요금제별 차이
    df.loc[df['요금제레벨'] == '고가', '월평균매출_ARPU'] *= 2.5
    df.loc[df['요금제레벨'] == '고가', '금융_접속건수'] *= 5.0
    
    # 단말유형별 차이
    df.loc[df['단말유형'] == '아이폰', 'SNS_접속건수'] *= 5.5
    df.loc[df['단말유형'] == '아이폰', '나이'] -= 25
    df.loc[df['단말유형'] == '갤럭시 프리미엄', '쇼핑_접속건수'] *= 3.5
    df.loc[df['단말유형'] == '키즈폰', '나이'] = np.random.randint(8, 14, len(df[df['단말유형']=='키즈폰']))

    # 후처리 (나이 범위 보정 등)
    df['나이'] = df['나이'].clip(8, 80)
    return df

# --- 3. 메인 UI ---
def main():
    st.title("🚀 통신 세그먼트 Deep-Dive 분석기 (Rich Output Ver.)")
    df = load_data()
    
    st.sidebar.header("📋 분석 설정")
    user_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    dim = st.selectbox("비교 기준(Dimension):", ["약정유형", "요금제레벨", "단말유형"])
    vals = df[dim].unique().tolist()
    col1, col2 = st.columns(2)
    with col1: g_a = st.selectbox("그룹 A", vals, index=0)
    with col2: g_b = st.selectbox("그룹 B", vals, index=1)

    if st.button("Deep-Dive 분석 시작", use_container_width=True):
        if g_a == g_b:
            st.error("⚠️ 서로 다른 두 그룹을 선택해주세요.")
            return

        df_a = df[df[dim] == g_a].copy()
        df_b = df[df[dim] == g_b].copy()
        
        # --- [1] IV 기반 Top 5 변수 선정 ---
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
                
                if 'IV' in bt.columns:
                    iv = bt.loc["전체", "IV"]
                    # 아주 미세한 차이라도 있으면 무조건 리스트에 포함
                    iv_list.append({'feature': col, 'iv': iv if iv > 0 else 0.001})
                    binning_tables[col] = bt
            except: continue
        
        # IV 순 정렬
        top_df = pd.DataFrame(iv_list).sort_values(by='iv', ascending=False)
        top_5 = top_df.head(5)
        
        # --- 결과 출력 ---
        st.header(f"📊 {g_a} vs {g_b} 핵심 차별점 랭킹")
        m_cols = st.columns(len(top_5))
        for i, row in enumerate(top_5.itertuples()):
            m_cols[i].metric(f"{i+1}위: {row.feature}", f"{row.iv:.3f}", "IV Score")

        st.divider()
        st.header("📈 상위 변수 분포 차이 (Half-Violin)")
        
        for feature in top_5['feature']:
            st.subheader(f"📍 {feature} 특성 비교")
            t1, t2 = st.columns([2, 1])
            with t1:
                fig = go.Figure()
                fig.add_trace(go.Violin(x=df_a[feature], name=g_a, side='negative', line_color='blue', meanline_visible=True))
                fig.add_trace(go.Violin(x=df_b[feature], name=g_b, side='positive', line_color='orange', meanline_visible=True))
                fig.update_layout(violingap=0, violinmode='overlay', height=350)
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                st.caption("구간별 데이터 비중 (OptBinning)")
                st.dataframe(binning_tables[feature][['Bin', 'Count', 'WoE', 'IV']].head(6), use_container_width=True)

        st.divider()
        st.header("📝 AI 전략 마케팅 제언")
        analysis_summary = f"상위 차이 변수: {top_5['feature'].tolist()}, IV 점수: {top_5['iv'].tolist()}"
        st.info(get_gemini_insight(f"{g_a}와 {g_b} 세그먼트 마케팅 전략", analysis_summary, user_api_key))

        st.divider()
        st.header("🔍 Sweetviz 상세 대조 리포트")
        report = sv.compare([df_a, g_a], [df_b, g_b])
        report.show_html("compare.html", open_browser=False)
        with open("compare.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
