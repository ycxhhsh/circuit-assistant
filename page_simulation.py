# --- filename: page_simulation.py ---
import streamlit as st
import streamlit.components.v1 as components

def show():
    # 注入 CSS 隐藏侧边栏和顶栏，开启沉浸式模式
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;} /* 隐藏侧边栏 */
            section.main > div {padding-top: 2rem;} /* 减少顶部留白 */
            #MainMenu {visibility: hidden;} /* 隐藏右上角菜单 */
            header {visibility: hidden;} /* 隐藏顶部装饰条 */
        </style>
        <div style='position: fixed; top: 10px; left: 10px; z-index:999;'>
            <a href='.' target='_self' style='background:#eee; padding:5px 10px; border-radius:5px; text-decoration:none;'>⬅️ 返回主页</a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; margin-bottom:0;'>🔌 交互式电路仿真</h2>", unsafe_allow_html=True)

    try:
        with open("simulation.html", "r", encoding='utf-8') as f:
            html_content = f.read()
        
        # 增加高度，height=1100 左右适合大部分 iPad 竖屏/横屏
        components.html(html_content, height=1100, scrolling=True)
        
    except FileNotFoundError:
        st.error("❌ 未找到 simulation.html 文件")
