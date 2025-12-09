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
        system_instruction = """
        你是一位专业的电子电路助教，正在指导学生连接 CD4026 计数器芯片。
        【核心引脚标准接法】：
        1. Pin 1 (CLK) -> 接时钟信号
        2. Pin 2 (INH) -> 接开关或接地
        3. Pin 3 (DEI) -> 接 VCC (Pin 16)
        4. Pin 15 (RST) -> 接接地 (Pin 8)
        请引导学生完成连接。
        """
        st.session_state.messages = [
            {"role": "system", "content": system_instruction}
        ]

def render_floating_assistant():
    """渲染平板优化的悬浮对话框 - 修复版"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* --- 1. 按钮容器：强制固定在右上角 --- */
    [data-testid="stPopover"] {
        position: fixed !important;
        top: 80px !important;       /* 距离顶部留出空间 */
        right: 40px !important;     /* 距离右侧留出空间 */
        left: auto !important;      /* 禁用左侧定位 */
        bottom: auto !important;
        z-index: 9999999 !important; /* 最高层级 */
        width: auto !important;     /* 防止容器撑满屏幕 */
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
        font-size: 32px !important; /* 图标大小 */
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

    /* --- 3. 弹出对话框：核心修复 --- */
    /* 强制对话框脱离文档流，固定在屏幕特定位置，防止被截断 */
    [data-testid="stPopoverBody"] {
        position: fixed !important;
        
        /* 这里的 top 值要等于：按钮top(80) + 按钮高度(64) + 间距(10) = 154px */
        top: 154px !important; 
        
        /* 强制靠右对齐，与按钮平齐 */
        right: 40px !important;
        left: auto !important;
        
        /* 禁用 Streamlit 的自动计算偏移，这是导致“四分五裂”的元凶 */
        transform: none !important; 
        
        width: 380px !important;
        max-width: 85vw !important; /* 防止手机上太宽 */
        max-height: 600px !important;
        
        border-radius: 12px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
        border: 1px solid #eee !important;
        z-index: 9999999 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 按钮内容
    with st.popover("🤖", use_container_width=False):
        st.markdown("### 💬 助教小电")
        
        # 聊天记录容器
        msg_container = st.container(height=350)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # 输入框
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
