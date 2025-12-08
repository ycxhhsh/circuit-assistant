# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

# 配置你的 API
API_KEY = "sk-vVIGbUylII5Kg9rZwGLZMzzubt778St90r66gGtTXTUs4shK" 
BASE_URL = "https://api.openai-proxy.org/v1"
MODEL_NAME = "gpt-3.5-turbo" 

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
            {"role": "system", "content": "你是一位专业的电子电路助教。"}
        ]

def render_floating_assistant():
    """渲染平板优化的悬浮对话框 (CSS 增强版)"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* 呼吸动画 */
    @keyframes pulse-purple {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }

    /* 1. 容器定位：强制右上角 */
    /* 注意：这里改用了更宽松的选择器，只要含有 stPopover 就可以 */
    [data-testid="stPopover"] {
        position: fixed !important;
        top: 30px !important;    /* 距离顶部调小一点，防遮挡 */
        right: 30px !important;  /* 距离右侧 */
        left: auto !important;   /* 必须强制取消左侧定位 */
        bottom: auto !important;
        z-index: 9999999 !important; /* 层级拉满 */
        transform: none !important;
        width: auto !important;
        height: auto !important;
    }
    
    /* 2. 按钮样式：大号紫色圆形 */
    /* 🔥 关键修改：把 "> div > button" 改成了 "button"，匹配更强 */
    [data-testid="stPopover"] button {
        width: 80px !important;
        height: 80px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 3px solid white !important; /* 加粗白边，更明显 */
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
        
        animation: pulse-purple 2s infinite;
        transition: transform 0.2s ease;
        
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 3. 图标样式 */
    [data-testid="stPopover"] button div,
    [data-testid="stPopover"] button span,
    [data-testid="stPopover"] button p {
        font-size: 40px !important;
        color: white !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 4. 交互效果 */
    [data-testid="stPopover"] button:active {
        transform: scale(0.9) !important;
        animation: none !important;
        background: #5a67d8 !important;
    }
    
    [data-testid="stPopover"] button:hover {
        transform: scale(1.05) !important;
    }

    /* 5. 展开后的对话框美化 */
    [data-testid="stPopoverBody"] {
        width: 400px !important;
        max-width: 90vw !important;
        border-radius: 20px !important;
        border: 1px solid #eee !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15) !important;
        top: 120px !important; /* 调整展开框的位置，不要盖住按钮 */
        right: 30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.popover("🤖", use_container_width=False):
        st.markdown("### 💬 助教小电")
        
        msg_container = st.container(height=400)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        if prompt := st.chat_input("我的电路哪里有问题？"):
            log_context = st.session_state.get("recognition_log", "（学生尚未上传图片）")
            dynamic_system_prompt = f"""
            你是一位专业的电子电路助教。
            【当前检测状态】：{log_context}
            请优先解答接线错误。
            """
            
            if len(st.session_state.messages) > 0 and st.session_state.messages[0]["role"] == "system":
                st.session_state.messages[0]["content"] = dynamic_system_prompt

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
