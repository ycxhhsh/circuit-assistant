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
        # --- 🔥 修改部分：更自然的人设 ---
        system_instruction = """
        你是一位友善、专业的电子电路助教（名字叫“助教小电”）。
        
        你的主要任务是解答学生关于电路、电子元器件（特别是 CD4026 芯片）以及实验调试的问题。
        
        【行为准则】
        1. 请像一位耐心的学长或老师一样正常交流，不要机械地重复规则。
        2. 只有当学生明确询问“怎么接线”、“引脚定义”或“电路连错了”时，你才需要引用具体的 CD4026 引脚知识（如 Pin 1 CLK, Pin 2 INH, Pin 15 RST 等）。
        3. 回答要简洁明了，鼓励学生自己动手尝试。
        """
        st.session_state.messages = [
            {"role": "system", "content": system_instruction}
        ]

def render_floating_assistant():
    """渲染平板优化的悬浮对话框 - 最终修复版"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* --- 1. 按钮容器：强制固定在右上角 --- */
    [data-testid="stPopover"] {
        position: fixed !important;
        top: 80px !important;       /* 避开顶部 Header */
        right: 40px !important;     /* 靠右 */
        left: auto !important;      /* 禁用左侧定位 */
        bottom: auto !important;
        z-index: 9999999 !important; /* 最高层级，防止被侧边栏遮挡 */
        width: auto !important;
    }

    /* --- 2. 按钮本体样式 --- */
    [data-testid="stPopover"] > div > button {
        width: 64px !important;
        height: 64px !important;
        border-radius: 50% !important;
        background: #ffffff !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        font-size: 32px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* 按下反馈 */
    [data-testid="stPopover"] > div > button:active {
        transform: scale(0.95);
        background-color: #f0f0f0 !important;
    }

    /* --- 3. 弹出对话框：强制固定位置，防止截断 --- */
    [data-testid="stPopoverBody"] {
        position: fixed !important;
        top: 154px !important;      /* 按钮底部下方 (80+64+10) */
        right: 40px !important;     /* 与按钮右对齐 */
        left: auto !important;
        transform: none !important; /* 关键：禁用 Streamlit 自动计算位置 */
        
        width: 380px !important;
        max-width: 85vw !important;
        max-height: 600px !important;
        
        border-radius: 12px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
        border: 1px solid #eee !important;
        z-index: 9999999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 按钮内容
    with st.popover("⚡", use_container_width=False):
        st.markdown("### 💬 助教小电")
        
        # 聊天记录容器
        msg_container = st.container(height=350)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # 输入框
        if prompt := st.chat_input("同学，有什么问题吗？"):
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
                                if chunk.choices:
                                    delta = chunk.choices[0].delta
                                    if delta.content:
                                        full_response += delta.content
                                        stream_box.markdown(full_response + "▌")
                            
                            stream_box.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.error("AI 客户端未初始化")
