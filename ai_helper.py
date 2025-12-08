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
    """渲染平板优化的悬浮对话框 (强制右上角版)"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* 定义呼吸动画：让按钮有“活着”的感觉 */
    @keyframes pulse-purple {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }

    /* 1. 定位容器：使用 !important 强制固定在右上角 */
    [data-testid="stPopover"] {
        position: fixed !important;
        top: 80px !important;    /* 距离顶部 */
        right: 40px !important;  /* 距离右侧 */
        left: auto !important;   /* 强制取消左侧定位 */
        z-index: 999999 !important;
        transform: none !important; /* 防止父容器干扰 */
    }
    
    /* 2. 按钮样式：强制变大 (80px) */
    [data-testid="stPopover"] > div > button {
        width: 80px !important;       /* 强制宽度 */
        height: 80px !important;      /* 强制高度 */
        border-radius: 50% !important; /* 强制圆形 */
        
        /* 渐变紫背景 */
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid white !important;
        
        /* 呼吸动画 */
        animation: pulse-purple 2s infinite;
        
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 3. 图标样式：强制变大 */
    [data-testid="stPopover"] > div > button > div,
    [data-testid="stPopover"] > div > button > span {
        font-size: 40px !important; /* 图标极大 */
        color: white !important;
        line-height: 1 !important;
    }
    
    /* 4. 按下反馈 */
    [data-testid="stPopover"] > div > button:active {
        transform: scale(0.9) !important;
        animation: none !important;
    }
    
    /* 5. 展开后的对话框美化 */
    [data-testid="stPopoverBody"] {
        width: 400px !important;
        max-width: 90vw !important;
        border-radius: 24px !important;
        border: none !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 按钮内容 (只放图标)
    with st.popover("🤖", use_container_width=False):
        st.markdown("### 💬 助教小电")
        
        # 消息容器
        msg_container = st.container(height=400)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # 输入框
        if prompt := st.chat_input("我的电路哪里有问题？"):
            # 获取最新判卷日志
            log_context = st.session_state.get("recognition_log", "（学生尚未上传图片或进行识别）")
            dynamic_system_prompt = f"""
            你是一位专业的电子电路助教。
            【当前检测状态】：{log_context}
            请优先解答接线错误。
            """
            
            # 更新 system prompt
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
