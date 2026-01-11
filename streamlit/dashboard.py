import streamlit as st
import json, os, time
from datetime import datetime

st.set_page_config(page_title="YOLO 采集指挥中心", layout="wide")

# --- 核心修复：向上跳一级定位根目录的 tasks.json ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR) 
JSON_PATH = os.path.join(BASE_DIR, "tasks.json")
# 注意：根据你的截图，图片似乎在根目录的 datasets/raw_signals
RAW_PATH = os.path.join(BASE_DIR, "datasets", "raw_signals")

# 侧边栏调试信息
st.sidebar.markdown(f"**文件状态检查**")
if os.path.exists(JSON_PATH):
    st.sidebar.success("🔗 tasks.json 已连接")
else:
    st.sidebar.error("❓ 未找到 tasks.json")
    st.sidebar.info(f"搜索路径: {JSON_PATH}")

st.title("📈 YOLO 实时信号采集流")

# 读取任务
tasks = []
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH, "r") as f:
            tasks = json.load(f)
    except: pass

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⏳ 任务调度倒计时")
    if not tasks:
        st.info("💡 暂时没有排队信号...")
    else:
        for t in tasks:
            run_at = datetime.strptime(t['run_at'], "%Y-%m-%d %H:%M:%S")
            rem = (run_at - datetime.now()).total_seconds()
            
            with st.container():
                st.markdown(f"""
                <div style="background:#161b22; padding:15px; border-radius:10px; border-left:5px solid #00ffcc; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:22px; font-weight:bold;">{t['symbol']}</span>
                        <span style="color:#ff4b4b;">倒计时: {int(max(0, rem))}s</span>
                    </div>
                    <div style="color:#8b949e; font-size:12px;">发出: {t['received_at']} | 计划: {t['run_at']}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, 1 - (rem / 600))))

with col2:
    st.subheader("📸 最新抓拍预览")
    if os.path.exists(RAW_PATH):
        files = sorted([f for f in os.listdir(RAW_PATH) if f.endswith('.png')], 
                       key=lambda x: os.path.getctime(os.path.join(RAW_PATH, x)), reverse=True)
        if files:
            st.image(os.path.join(RAW_PATH, files[0]), caption=f"最新: {files[0]}")
        else:
            st.warning("暂无图片预览")
    else:
        st.error(f"路径不存在: {RAW_PATH}")

time.sleep(3)
st.rerun()