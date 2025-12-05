import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import json

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="智能电路助教小电",
    page_icon="⚡️",
    layout="wide"
)

# ================= 2. 侧边栏：小电的调试面板 =================
st.sidebar.title("⚙️ 小电调试台")
st.sidebar.info("如果觉得小电眼睛不准，可以在这里微调：")

# 默认参数：自信度 0.10，判定距离 60 (宽松一点，防止误判)
CONF_THRESHOLD = st.sidebar.slider("AI 自信度 (Conf)", 0.05, 1.0, 0.10, 0.05)
DIST_THRESHOLD = st.sidebar.slider("判定距离 (Pixel)", 20, 100, 60, 5)

# ================= 3. 核心功能函数 =================

@st.cache_resource
def load_resources():
    """加载模型、基准图和坐标"""
    try:
        model = YOLO('best.pt')
        base_img = cv2.imread('base_fixed.jpg')
        if base_img is None: return None, None, None, "❌ 找不到 base_fixed.jpg"
        
        with open('board_config_fixed.json', 'r', encoding='utf-8') as f:
            pin_coords = json.load(f)
            
        return model, base_img, pin_coords, "OK"
    except Exception as e:
        return None, None, None, str(e)

# 初始化资源
model, base_img, pin_coords, msg = load_resources()
if msg != "OK": st.error(f"小电启动失败: {msg}"); st.stop()

# 准备备用对齐特征 (ORB)
orb = cv2.ORB_create(nfeatures=5000)
kp_base, des_base = orb.detectAndCompute(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), None)
h_ref, w_ref = base_img.shape[:2]

def check_color_in_roi(img, x, y, radius, target_color):
    """
    🚑 急救包：如果 YOLO 没看清，小电用放大镜去找颜色
    """
    x, y, r = int(x), int(y), int(radius)
    h, w = img.shape[:2]
    # 边界检查
    y1, y2 = max(0, y-r), min(h, y+r)
    x1, x2 = max(0, x-r), min(w, x+r)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return False

    # 转 HSV 颜色空间
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    
    if target_color == 'red':
        # 红色跨越 0度 和 180度
        lower1 = np.array([0, 100, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 100, 80])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        
    elif target_color == 'black':
        # 黑色看亮度(V)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 60]) # V < 60 算黑
        mask = cv2.inRange(hsv, lower, upper)

    # 只要区域内有 10% 是目标颜色，就算找到了
    if mask is not None:
        ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
        return ratio > 0.10 
    return False

def order_points(pts):
    """ 辅助函数：整理四个角点顺序 """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def align_image(raw_img):
    """ 双保险对齐：优先找绿板子，找不到再找特征点 """
    # --- 策略A: 颜色轮廓 (Green Contour) ---
    try:
        hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
        # 提取绿色
        mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 20000: # 面积够大才算
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    dst = np.array([[0,0], [w_ref-1,0], [w_ref-1,h_ref-1], [0,h_ref-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(order_points(approx.reshape(4,2)), dst)
                    return cv2.warpPerspective(raw_img, M, (w_ref, h_ref))
    except: pass

    # --- 策略B: 特征点 (ORB) ---
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None: return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_base, des), key=lambda x: x.distance)
    good = matches[:int(len(matches)*0.15)]
    if len(good) < 4: return None
    
    src_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_base[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(raw_img, M, (w_ref, h_ref)) if M is not None else None

# ================= 4. 主界面逻辑 =================
st.title("⚡️ 智能电路助教小电")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 1. 上传作业")
    uploaded_file = st.file_uploader("请把照片拖进来", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    with col1: st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), caption="学生作业原图", use_column_width=True)

    # --- 开始分析 ---
    with col2:
        st.subheader("📝 2. 小电批改结果")
        
        aligned_img = align_image(raw_img)
        if aligned_img is None:
            st.error("⚠️ 小电看不清电路板！请确保拍摄清晰，建议下面垫一张白纸。")
        else:
            # 1. YOLO 初筛
            results = model(aligned_img, conf=CONF_THRESHOLD, iou=0.8, verbose=False)[0]
            detected_heads = [{"color": model.names[int(b.cls[0])], "x": b.xywh[0][0].item(), "y": b.xywh[0][1].item()} for b in results.boxes]

            # 2. 坐标归位 (支持一孔多线)
            board_status = {}
            for pin, (px, py) in pin_coords.items():
                board_status[pin] = {"connected": False, "colors": []}
                for head in detected_heads:
                    if math.sqrt((head['x']-px)**2 + (head['y']-py)**2) < DIST_THRESHOLD:
                        board_status[pin]["connected"] = True
                        board_status[pin]["colors"].append(head['color'])

            # 3. 判卷 + 急救包逻辑
            errors, praises = [], []

            # --- 规则1: VCC (红色) ---
            vcc_pin = "U1_Pin_16 (VCC)"
            vcc = board_status[vcc_pin]
            px, py = pin_coords[vcc_pin]
            
            # 补丁：如果没连，或者没检测到红线，启动颜色强侦测
            if not vcc['connected'] or "head_red" not in vcc['colors']:
                if check_color_in_roi(aligned_img, px, py, 20, 'red'):
                    vcc['connected'] = True
                    if "head_red" not in vcc['colors']: vcc['colors'].append("head_red")
            
            if not vcc['connected']:
                errors.append("❌ **芯片没供电**：U1 Pin 16 未连接。")
            elif "head_red" in vcc['colors']:
                praises.append("电源 VCC 连接正确 (红色)")
            else:
                errors.append(f"⚠️ **颜色不规范**：VCC 建议红线，小电检测到 {vcc['colors']}。")

            # --- 规则2: GND (黑色) ---
            gnd_pin = "U1_Pin_8 (GND)"
            gnd = board_status[gnd_pin]
            gx, gy = pin_coords[gnd_pin]
            
            # 补丁：启动黑色强侦测
            if not gnd['connected'] or "head_black" not in gnd['colors']:
                if check_color_in_roi(aligned_img, gx, gy, 20, 'black'):
                    gnd['connected'] = True
                    if "head_black" not in gnd['colors']: gnd['colors'].append("head_black")

            if not gnd['connected']:
                errors.append("❌ **芯片没接地**：U1 Pin 8 未连接。")
            elif "head_black" in gnd['colors']:
                praises.append("接地 GND 连接正确 (黑色)")
            else:
                errors.append(f"⚠️ **颜色不规范**：GND 建议黑线，小电检测到 {gnd['colors']}。")

            # --- 规则3: 数码管 ---
            seg_pins = ["U1_Pin_6 (Seg F)", "U1_Pin_7 (Seg G)", "U1_Pin_9 (Seg D)", "U1_Pin_10 (Seg A)", "U1_Pin_11 (Seg E)", "U1_Pin_12 (Seg B)", "U1_Pin_13 (Seg C)"]
            conn_count = sum(1 for p in seg_pins if board_status[p]['connected'])
            if conn_count < 7: errors.append(f"❌ **数码管缺笔画**：只接了 {conn_count}/7 根。")
            else: praises.append(f"数码管 7 段驱动线完整")

            # 显示结果
            if errors: 
                st.warning(f"小电发现 {len(errors)} 个问题，请修改：")
                for e in errors: st.markdown(e)
            else: 
                st.success("🎉 完美！电路连接正确！小电给满分！"); st.balloons()
            
            if praises:
                with st.expander("👀 查看做得好的地方"):
                    for p in praises: st.write(f"✅ {p}")

            # 画图 (蓝色=判定区，绿色=识别点)
            viz = aligned_img.copy()
            for p, (px, py) in pin_coords.items():
                cv2.circle(viz, (int(px), int(py)), DIST_THRESHOLD, (255, 0, 0), 2) # 判定圈
            for h in detected_heads:
                cv2.circle(viz, (int(h['x']), int(h['y'])), 6, (0, 255, 0), -1) # 识别点
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="小电视觉分析图 (蓝圈=判定区)", use_column_width=True)

else:
    st.info("👈 请在左侧上传图片，小电随时准备着！")