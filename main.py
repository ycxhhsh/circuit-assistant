# --- filename: main.py ---
import streamlit as st
import ai_helper
import page_recognition
import page_simulation

# 1. 全局页面配置 (必须放在第一行)
st.set_page_config(page_title="智能电路助教平台", page_icon="⚡️", layout="wide")

# 2. 渲染 AI 助手 (所有页面通用)
ai_helper.render_floating_assistant()

# 3. 侧边栏导航
st.sidebar.title("🚀 导航")
page = st.sidebar.radio("选择功能模块", ["📷 拍照判卷", "🔌 仿真实验"])

# 4. 页面路由
if page == "📷 拍照判卷":
    page_recognition.show()
elif page == "🔌 仿真实验":
    page_simulation.show()