# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

# 配置你的 API (保持不变)
API_KEY = "sk-vVIGbUylII5Kg9rZwGLZMzzubt778St90r66gGtTXTUs4shK" 
BASE_URL = "https://api.openai-proxy.org/v1"
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
        # 🔥 修改点：设置为通用电子助教，不再绑定具体实验
        st.session_state.messages = [
            {
                "role": "system", 
                "content": (
                    "你是一位专业的电子电路助教 '小电'。"
                    "你的职责是解答学生关于电子电路、元器件原理、仪器使用（如万用表、示波器）、"
                    "焊接安全以及故障排查的一般性问题。"
                    "你的语气要活泼、鼓励，适合中职或职高学生。"
                    "如果学生问到具体实验步骤，你可以给出通用的指导，但不需要针对特定的 CD4026 连线进行评分。"
                )
            }
        ]

def render_floating_assistant():
    """渲染平板优化的悬浮对话框"""
    init_ai_session()
    
    # CSS 样式保持不变，维持良好的触控体验
    st.markdown("""
    <style>
    /* 悬浮球位置 */
    [data-testid="stPopover"] {
        position: fixed;
        top: 100px; 
        right: 30px;
        z-index: 99999;
    }
    
    /* 悬浮球按钮样式 */
    [data-testid="stPopover"] > div > button {
        width: 72px; height: 72px; border-radius: 35px;
        background: #ffffff; color: #333; border: 1px solid #e0e0e0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12); 
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex; align-items: center; justify-content: center; padding: 0;
    }
    [data-testid="stPopover"] > div > button > div { font-size: 36px !important; }
    
    /* 点击反馈 */
    [data-testid="stPopover"] > div > button:active {
        transform: scale(0.9); background-color: #f5f5f5;
    }
    
    /* 聊天窗口样式 */
    [data-testid="stPopoverBody"] {
        width: 380px !important; max-width: 90vw;
        border-radius: 20px !important; border: none !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 渲染悬浮按钮
    with st.popover("⚡", use_container_width=False):
        st.markdown("### 💬 助教小电")
        st.caption("我是你的电子实验小助手，有什么问题都可以问我！")
        
        # 消息容器
        msg_container = st.container(height=400)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # 输入框
        if prompt := st.chat_input("比如：数码管为什么不亮？"):
            # 直接处理用户输入，不再注入 Context
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
                            for chunk in stream:
                                if chunk.choices and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    if delta.content:
                                        full_response += delta.content
                                        stream_box.markdown(full_response + "▌")
                            
                            stream_box.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"AI 响应中断: {str(e)}")
            else:
                st.error("AI 客户端未初始化")
