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
        # 设定一个静态的 System Prompt，包含电路知识，但不包含实时检测状态
        system_instruction = """
        你是一位专业的电子电路助教，正在指导学生连接 CD4026 计数器芯片。

        【实验背景信息】
        为了降低难度，数码管和电源线通常已预设接好。
        学生主要负责以下 4 个核心引脚的连接，标准接法如下：
        1. Pin 1 (CLK) -> 接时钟信号
        2. Pin 2 (INH) -> 接开关或接地 (低电平有效)
        3. Pin 3 (DEI) -> 接 VCC (Pin 16) (高电平有效)
        4. Pin 15 (RST) -> 接接地 (Pin 8) (复位端)

        请根据以上标准回答学生的提问。如果学生问“我该怎么接”，请引导他们完成这四个引脚的连接。
        """
        st.session_state.messages = [
            {"role": "system", "content": system_instruction}
        ]

def render_floating_assistant():
    """渲染平板优化的悬浮对话框"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* 1. 定位容器：右上角偏下 */
    [data-testid="stPopover"] {
        position: fixed;
        top: 100px;
        right: 30px;
        z-index: 99999;
    }
    
    /* 2. 按钮样式：大号平板触控版 */
    [data-testid="stPopover"] > div > button {
        width: 72px;
        height: 72px;
        border-radius: 35px;
        background: #ffffff;
        color: #333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12); 
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
    }
    
    /* 3. 放大内部的 Emoji 图标 */
    [data-testid="stPopover"] > div > button > div {
        font-size: 36px !important;
    }
    
    /* 4. 按下效果 */
    [data-testid="stPopover"] > div > button:active {
        transform: scale(0.9);
        background-color: #f5f5f5;
    }
    
    /* 5. 展开后的对话框样式 */
    [data-testid="stPopoverBody"] {
        width: 380px !important;
        max-width: 90vw;
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 按钮里只放一个图标
    with st.popover("🤖", use_container_width=False):
        st.markdown("### 💬 助教小电")
        
        msg_container = st.container(height=400)
        with msg_container:
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # 提示语可以改得通用一点
        if prompt := st.chat_input("关于电路有什么问题？"):
            
            # --- 变动处：移除了之前的 log_context 获取和 system prompt 动态更新逻辑 ---
            
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
