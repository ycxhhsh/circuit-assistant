# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

# 配置你的 API
API_KEY = "sk-vVIGbUylII5Kg9rZwGLZMzzubt778St90r66gGtTXTUs4shK" 
BASE_URL = "https://api.openai-proxy.org/v1"
# 建议先用 gpt-3.5-turbo 测试，因为它最稳定。确认能用后再改回 gpt-4o
MODEL_NAME = "gpt-4o" 

def init_ai_session():
    """初始化 AI 客户端和历史记录"""
    if "ai_client" not in st.session_state:
        try:
            st.session_state.ai_client = OpenAI(
                api_key=API_KEY, 
                base_url=BASE_URL
            )
        except Exception as e:
            st.error(f"AI 客户端初始化失败: {e}")
            st.session_state.ai_client = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "你是一位专业的电子电路助教，负责解答学生关于CD4026芯片、数码管接线和数字电路的问题。回答要简洁、准确。"}
        ]

def render_floating_assistant():
    """渲染底部长条形悬浮对话框"""
    init_ai_session()
    
    # CSS 样式保持不变
    st.markdown("""
    <style>
    [data-testid="stPopover"] {
        position: fixed;
        bottom: 40px;
        right: 40px; 
        z-index: 9999;
    }
    [data-testid="stPopover"] > div > button {
        width: 260px;  
        height: 55px;
        border-radius: 30px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white;
        border: none;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stPopover"] > div > button::after {
        content: "🤖 有问题？问问 AI 助教";
        margin-left: 8px;
    }
    [data-testid="stPopover"] > div > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.popover("💬", use_container_width=False):
        st.markdown("### 🤖 电路百事通")
        
        msg_container = st.container(height=400)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        if prompt := st.chat_input("输入问题..."):
            with msg_container:
                st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if st.session_state.ai_client:
                with msg_container:
                    with st.chat_message("assistant"):
                        stream_box = st.empty()
                        full_response = ""
                        try:
                            stream = st.session_state.ai_client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=st.session_state.messages,
                                stream=True
                            )
                            # 🔥 修复重点：增加安全检查逻辑
                            for chunk in stream:
                                # 1. 检查 choices 列表是否存在且不为空
                                if chunk.choices and len(chunk.choices) > 0:
                                    # 2. 只有当 delta.content 存在时才拼接
                                    delta = chunk.choices[0].delta
                                    if delta.content:
                                        full_response += delta.content
                                        stream_box.markdown(full_response + "▌")
                            
                            stream_box.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            # 打印更详细的错误方便排查
                            st.error(f"AI 响应中断: {str(e)}")
            else:
                st.error("AI 客户端未初始化")