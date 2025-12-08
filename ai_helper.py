# --- filename: ai_helper.py ---
import streamlit as st
from openai import OpenAI

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
    """渲染平板优化的悬浮对话框"""
    init_ai_session()
    
    st.markdown("""
    <style>
    /* 1. 定位容器：为了平板好按，建议放在右下角或者右上角偏下的位置 */
    /* 这里设定为：右上角，但往下挪一点，避开平板的状态栏和菜单 */
    [data-testid="stPopover"] {
        position: fixed;
        top: 100px;       /* 距离顶部 100px */
        right: 30px;      /* 距离右侧 30px */
        z-index: 99999;
    }
    
    /* 2. 按钮样式：大号平板触控版 */
    [data-testid="stPopover"] > div > button {
        width: 72px;        /* 增大宽度 */
        height: 72px;       /* 增大高度 */
        border-radius: 35px; /* 保持圆形 (高度的一半) */
        background: #ffffff;
        color: #333;
        border: 1px solid #e0e0e0;
        /* 更深的阴影，制造悬浮感 */
        box-shadow: 0 8px 24px rgba(0,0,0,0.12); 
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
    }
    
    /* 3. 放大内部的 Emoji 图标 */
    [data-testid="stPopover"] > div > button > div {
        font-size: 36px !important; /* 图标放大 */
    }
    
    /* 4. 按下效果 (Active) - 模拟真实按钮反馈 */
    [data-testid="stPopover"] > div > button:active {
        transform: scale(0.9);
        background-color: #f5f5f5;
    }
    
    /* 5. 展开后的对话框样式 */
    [data-testid="stPopoverBody"] {
        width: 380px !important; /* 对话框也可以宽一点 */
        max-width: 90vw; /* 防止超出手机屏幕 */
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
