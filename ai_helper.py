# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

# 配置你的 API
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
        # 初始化 system prompt，稍后我们会动态更新它
        st.session_state.messages = [
            {"role": "system", "content": "你是一位专业的电子电路助教。"}
        ]

def render_floating_assistant():
    """渲染底部长条形悬浮对话框"""
    init_ai_session()
    
    # CSS 样式 (保持不变)
    st.markdown("""
    <style>
    [data-testid="stPopover"] {
        position: fixed; bottom: 40px; right: 40px; z-index: 9999;
    }
    [data-testid="stPopover"] > div > button {
        width: 260px; height: 55px; border-radius: 30px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; border: none; box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        font-size: 16px; font-weight: bold; transition: all 0.3s ease;
        display: flex; align-items: center; justify-content: center;
    }
    [data-testid="stPopover"] > div > button::after {
        content: "🤖 有问题？问问 AI 助教"; margin-left: 8px;
    }
    [data-testid="stPopover"] > div > button:hover {
        transform: translateY(-5px); box-shadow: 0 15px 25px rgba(0,0,0,0.3);
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

        if prompt := st.chat_input("我的电路哪里有问题？"):
            # ================= 🔥 核心优化：动态注入上下文 =================
            # 1. 获取最新的判卷日志
            log_context = st.session_state.get("recognition_log", "（学生尚未上传图片或进行识别）")
            
            # 2. 构造更加智能的 System Prompt
            dynamic_system_prompt = f"""
            你是一位专业的电子电路助教，负责指导学生连接 CD4026 计数器电路。
            
            【当前学生的电路板状态（由视觉算法检测）】
            {log_context}
            
            请根据检测到的错误（如果有），优先解答学生的接线问题。
            如果检测报告全是正确的，请夸奖学生。
            回答要亲切、简洁，不要长篇大论。
            """
            
            # 3. 悄悄更新 system prompt (messages[0])，让 AI 知道最新情况
            if len(st.session_state.messages) > 0 and st.session_state.messages[0]["role"] == "system":
                st.session_state.messages[0]["content"] = dynamic_system_prompt
            # ================= 🔥 优化结束 =================

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