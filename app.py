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

# --- 2. 무조건 차이가 나도록 설계된 가상 데이터 ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1500
    약정유형_list = ['SIM-only', '단말약정']
    요금제_list = ['고가', '중가', '저가']
    단말_list = ['아이폰', '갤럭시 프리미엄', '갤럭시 중저가', '키즈폰']
    
    data = {
        '고객ID': range(10001, 10001 + n),
        '약정유형': np.random.choice(약정유형_list, n),
        '요금제레벨': np.random.choice(요금제_list, n),
        '단말유형': np.random.choice(단말_list, n),
        '나이': np.random.randint(20, 70, n),
        '월평균매출_ARPU': np.random.randint(30000, 80000, n).astype(float),
        'OTT_접속건수': np.random.randint(5, 100, n).astype(float),
        '쇼핑_접속건수': np.random.randint(5, 80, n).astype(float),
        'SNS_접속건수': np.random.randint(10, 150, n).astype(float),
        '금융_접속건수': np.random.randint(2, 40, n).astype(float),
        '데이터사용량_GB': np.random.uniform(10, 50, n)
    }
    df = pd.DataFrame(data)
    
    # 강력한 편향 주입
    df.loc[df['약정유형'] == 'SIM-only', 'OTT_접속건수'] *= 5.0
    df.loc[df['약정유형'] == 'SIM-only', '데이터사용량_GB'] *= 3.0
    df.loc[df['요금제레벨'] == '고가', '월평균매출_ARPU'] *= 2.5
    df.loc[df['요금제레벨'] == '고가', '금융_접속건수'] *= 5.0
    df.loc[df['단말유형'] == '아이폰', 'SNS_접속건수'] *= 5.0
    df.loc[df['단말유형'] == '아이폰', '나이'] -= 20
    
    df['나이'] = df['나이'].clip(8, 80)
    return df

# --- 3. 메인 UI ---
def main():
    st.title("🚀 통신 세그먼트 Deep-Dive 분석기 (Crash-Free Ver.)")
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
            st.warning("⚠️ 서로 다른 두 그룹을 선택해주세요.")
            return

        df_a = df[df[dim] == g_a].copy()
        df_b = df[df[dim] == g_b].copy()
        
        # --- [1] IV 계산 및 에러 핸들링 ---
        temp_a = df_a.copy(); temp_a['target'] = 0
        temp_b = df_b.copy(); temp_b['target'] = 1
        combined = pd.concat([temp_a, temp_b])
        
        features = ['나이', '월평균매출_ARPU', 'OTT_접속건수', '쇼핑_접속건수', 'SNS_접속건수', '금융_접속건수', '데이터사용량_GB']
        iv_list = []
        binning_tables = {}

        for col in features:
            try:
                optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
                optb.fit(combined[col].values, combined['target'].values)
                bt = optb.binning_table.build()
                iv_val = bt.loc["전체", "IV"]
                
                # 정상적인 IV 값만 저장
                if pd.notnull(iv_val):
                    iv_list.append({'feature': col, 'iv': float(iv_val)})
                    binning_tables[col] = bt
            except:
                continue
        
        # 만약 iv_list가 비어있다면 강제로 더미 데이터 생성 (KeyError 방지)
        if not iv_list:
            top_5 = pd.DataFrame([{'feature': f, 'iv': 0.0} for f in features[:5]])
        else:
            top_5 = pd.DataFrame(iv_list).sort_values(by='iv', ascending=False).head(5)

        # --- [2] 결과 렌더링 ---
        st.header(f"📊 {g_a} vs {g_b} 핵심 차별점 랭킹")
        m_cols = st.columns(len(top_5))
        for i, row in enumerate(top_5.itertuples()):
            m_cols[i].metric(f"{i+1}위: {row.feature}", f"{row.iv:.3f}")

        st.divider()
        st.header("📈 상위 변수 분포 차이 (Half-Violin)")
        
        for feature in top_5['feature']:
            if feature in binning_tables or feature in df.columns:
                st.subheader(f"📍 {feature} 특성 분석")
                t1, t2 = st.columns([2, 1])
                with t1:
                    fig = go.Figure()
                    fig.add_trace(go.Violin(x=df_a[feature], name=g_a, side='negative', line_color='blue'))
                    fig.add_trace(go.Violin(x=df_b[feature], name=g_b, side='positive', line_color='orange'))
                    fig.update_layout(violingap=0, violinmode='overlay', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with t2:
                    if feature in binning_tables:
                        st.dataframe(binning_tables[feature][['Bin', 'Count', 'WoE', 'IV']].head(5))
                    else:
                        st.write("상세 구간 데이터 없음")

        # --- [3] AI & Sweetviz ---
        st.divider()
        st.header("📝 AI 전략 제언")
        st.info(get_gemini_insight("세그먼트 전략", f"Top 변수: {top_5['feature'].tolist()}", user_api_key))

        st.divider()
        st.header("🔍 상세 데이터 리포트 (Sweetviz)")
        report = sv.compare([df_a, g_a], [df_b, g_b])
        report.show_html("compare.html", open_browser=False)
        with open("compare.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
