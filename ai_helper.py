# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

# 配置你的 API
API_KEY = "sk-vVIGbUylII5Kg9rZwGLZMzzubt778St90r66gGtTXTUs4shK" 
BASE_URL = "https://api.openai-proxy.org/v1"
MODEL_NAME = "gpt-4o" 

def init_ai_session():
    if "ai_client" not in st.session_state:
        try:
            st.session_state.ai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        except:
            st.session_state.ai_client = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            # 🔥 这里的 System Prompt 改回通用版本，不需要读取判卷结果
            {"role": "system", "content": "你是一位幽默风趣的电子电路助教，可以回答学生关于 CD4026 芯片和数字电路的问题。"}
        ]

def render_floating_assistant():
    init_ai_session()
    
    # 样式保持不变...
    st.markdown("""
    <style>
    [data-testid="stPopover"] { position: fixed; top: 100px; right: 30px; z-index: 99999; }
    [data-testid="stPopover"] > div > button { width: 72px; height: 72px; border-radius: 35px; box-shadow: 0 8px 24px rgba(0,0,0,0.12); font-size: 36px !important; }
    [data-testid="stPopoverBody"] { width: 380px !important; max-width: 90vw; border-radius: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.popover("🤖", use_container_width=False):
        st.markdown("### 💬 助教小电")
        msg_container = st.container(height=400)
        
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("有问题随时问我..."):
            with msg_container: st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if st.session_state.ai_client:
                try:
                    stream = st.session_state.ai_client.chat.completions.create(
                        model=MODEL_NAME, messages=st.session_state.messages, stream=True
                    )
                    with msg_container:
                        with st.chat_message("assistant"):
                            resp = st.write_stream(stream)
                            st.session_state.messages.append({"role": "assistant", "content": resp})
                except: st.error("AI 响应异常")
