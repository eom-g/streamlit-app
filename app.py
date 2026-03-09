import os
import subprocess
import sys

# 1. pkg_resources 에러 강제 해결
try:
    import pkg_resources
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import json
import re
import google.generativeai as genai
from optbinning import OptimalBinning
from ydata_profiling import ProfileReport
import sweetviz as sv
import streamlit.components.v1 as components
import plotly.express as px

# --- 0. 호환성 패치 ---
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (Warning,), {})

st.set_page_config(page_title="The Ultimate EDA Agent", layout="wide")

# --- 1. API 설정 ---
with st.sidebar:
    st.header("🔑 API 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    found_model = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for p in ["flash", "1.5-pro"]:
                for m in available:
                    if p in m.lower():
                        found_model = m
                        break
                if found_model: break
            if found_model: st.success(f"✅ 모델 연결됨")
        except Exception as e: st.error(f"❌ 연결 오류: {e}")

# --- 2. 데이터 시뮬레이터 ---
def generate_realistic_data(plan):
    n = 1000
    t_name, r_name = plan['target_group_name'], plan['reference_group_name']
    features = plan['selected_features']
    labels = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    df = pd.DataFrame({'group': labels, '비교그룹': np.where(labels == 1, t_name, r_name)})
    
    for f in features:
        if "age" in f.lower():
            df[f] = np.random.randint(20, 70, n)
            df.loc[df['group']==1, f] = np.random.randint(20, 38, sum(labels))
        elif any(kw in f.lower() for kw in ["usage", "hrs", "amt", "time"]):
            df[f] = np.random.exponential(100, n)
            df.loc[df['group']==1, f] *= 2.6
        else:
            df[f] = np.random.normal(100, 30, n)
            df.loc[df['group']==1, f] += 65
    return df

# --- 3. 메인 분석 ---
st.title("🏆 The Ultimate Strategic EDA Agent")
user_hypo = st.text_area("사업 가설 입력:", "주말 야간에 쇼핑 앱 접속이 잦은 30대 여성 타겟 분석", height=80)

if st.button("🚀 전체 분석 리포트 생성", type="primary"):
    if not api_key or not found_model:
        st.warning("API Key를 먼저 설정해주세요.")
    else:
        with st.status("AI가 모든 분석 도구를 가동 중입니다...") as status:
            # [1] LLM 설계
            model = genai.GenerativeModel(found_model)
            prompt = f"가설 분석 JSON 응답: {user_hypo}. 피처풀: [age, data_usage, youtube_hrs, delivery_amt, night_usage, online_shop_amt, roaming_days]. 형식: {{\"target_group_name\":\"\", \"reference_group_name\":\"\", \"selected_features\":[]}}"
            res = model.generate_content(prompt)
            plan = json.loads(re.search(r'\{.*\}', res.text, re.DOTALL).group())
            
            df = generate_realistic_data(plan)
            
            # [2] Optbinning 분석 (IV 산출)
            iv_list = []
            for f in plan['selected_features']:
                try:
                    optb = OptimalBinning(name=f, dtype="numerical")
                    optb.fit(df[f], df['group'])
                    optb.binning_table.build()
                    iv_list.append({"지표": f, "IV(중요도)": optb.binning_table.iv})
                except: continue
            iv_df = pd.DataFrame(iv_list).sort_values(by="IV(중요도)", ascending=False)

            # [3] 줄글 인사이트
            stats_sum = "\n".join([f"- {f}: {plan['target_group_name']}({df[df['group']==1][f].mean():.1f}) vs {plan['reference_group_name']}({df[df['group']==0][f].mean():.1f})" for f in plan['selected_features']])
            insight_text = model.generate_content(f"사업팀 보고용 인사이트 요약 작성: {user_hypo}\n결과:\n{stats_sum}").text

            # --- 출력 영역 ---
            st.subheader("📝 Executive Summary")
            st.info(insight_text)

            col1, col2 = st.columns([2, 3])
            with col1:
                st.subheader("🎯 지표별 변별력 (IV)")
                st.dataframe(iv_df, use_container_width=True)
                st.caption("IV 0.3 이상: 강력한 변별력 | 0.1~0.3: 중간 | 0.02~0.1: 약함")
            with col2:
                top_feat = iv_df.iloc[0]['지표']
                st.plotly_chart(px.violin(df, x="비교그룹", y=top_feat, color="비교그룹", box=True, title=f"가장 중요한 지표: {top_feat}"), use_container_width=True)

            st.divider()
            tab1, tab2 = st.tabs(["📊 상세 대조 (Sweetviz)", "🩺 전체 건강진단 (Profiling)"])
            
            with tab1:
                report = sv.compare([df[df['group']==1], plan['target_group_name']], [df[df['group']==0], plan['reference_group_name']])
                report.show_html(filepath="sv.html", open_browser=False)
                with open("sv.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=900, scrolling=True)
            
            with tab2:
                profile = ProfileReport(df, minimal=True)
                components.html(profile.to_html(), height=900, scrolling=True)

            status.update(label="종합 보고서 생성 완료!", state="complete")
