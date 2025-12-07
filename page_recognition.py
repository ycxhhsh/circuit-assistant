# --- filename: page_recognition.py ---
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import copy

# ================= 资源加载与缓存 =================
@st.cache_resource
def load_resources():
    """
    加载模型、底图、坐标配置，并预计算底图的 ORB 特征点
    """
    try:
        model = YOLO('best.pt') 
        base_img = cv2.imread('base_fixed.jpg')
        if base_img is None: return None, None, None, None, "❌ 找不到 base_fixed.jpg"
        
        with open('board_config_fixed.json', 'r', encoding='utf-8') as f:
            pin_coords = json.load(f)
            
        # 初始化 ORB 并计算基准图特征 (用于后续对齐)
        orb = cv2.ORB_create(nfeatures=5000)
        kp_ref, des_ref = orb.detectAndCompute(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), None)
        
        return model, base_img, pin_coords, (orb, kp_ref, des_ref), "OK"
    except Exception as e:
        return None, None, None, None, str(e)

# ================= 核心图像算法 (来自你的 web_app.py) =================
PADDING = 40  # 全局填充

def correct_orientation(img):
    h, w = img.shape[:2]
    if h > w:
        st.toast("📷 检测到竖向拍摄，正在自动旋转...")
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def check_color_in_roi(img, x, y, radius, target_color):
    """颜色急救检查"""
    x, y, r = int(x), int(y), int(radius)
    h, w = img.shape[:2]
    roi = img[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
    if roi.size == 0: return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    if 'red' in target_color:
        mask = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([160, 80, 50]), np.array([180, 255, 255]))
    elif 'black' in target_color:
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    
    return (cv2.countNonZero(mask) / roi.size > 0.05) if mask is not None else False

def get_dominant_color(img, x, y, radius=25):
    """获取区域主导颜色"""
    x, y, r = int(x), int(y), int(radius)
    h, w = img.shape[:2]
    roi = img[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
    if roi.size == 0: return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'red': [(np.array([0, 80, 50]), np.array([10, 255, 255])), (np.array([160, 80, 50]), np.array([180, 255, 255]))],
        'yellow': [(np.array([20, 80, 80]), np.array([35, 255, 255]))],
        'green': [(np.array([35, 40, 40]), np.array([85, 255, 255]))],
        'cyan': [(np.array([85, 80, 80]), np.array([100, 255, 255]))],
        'blue': [(np.array([100, 60, 40]), np.array([130, 255, 255]))],
        'purple': [(np.array([125, 40, 40]), np.array([155, 255, 255]))],
        'black': [(np.array([0, 0, 0]), np.array([180, 255, 60]))] 
    }
    max_pixels, best_color = 0, None
    for color_name, bounds in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for (lower, upper) in bounds: mask += cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        if count > (roi.size * 0.05) and count > max_pixels: max_pixels = count; best_color = color_name
    return best_color

def align_image_robust(raw_img, base_img, orb_data):
    """
    强鲁棒性对齐算法：优先尝试轮廓透视变换，失败则回退到特征点匹配
    """
    orb, kp_ref, des_ref = orb_data
    h_ref, w_ref = base_img.shape[:2]
    img = correct_orientation(raw_img)
    h_img, w_img = img.shape[:2]
    
    # 目标尺寸 (加 padding)
    w_new, h_new = w_ref + 2 * PADDING, h_ref + 2 * PADDING

    # 内部函数：执行透视变换
    def warp_with_padding(src_pts):
        dst_pts = np.float32([[PADDING, PADDING], [PADDING, h_ref + PADDING], 
                              [w_ref + PADDING, h_ref + PADDING], [w_ref + PADDING, PADDING]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, (w_new, h_new))

    # 策略 1: 基于黄绿色底板的轮廓提取 (速度快，效果好)
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 针对绿色电路板的 HSV 范围
        mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > (h_img * w_img * 0.3): # 面积要够大
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
                # 排序四个角点：左上，左下，右下，右上 (大概顺序，需细调，这里简化处理)
                # 简单排序法：sum(x+y)最小是左上，最大是右下
                s = box.sum(axis=1)
                tl = box[np.argmin(s)]
                br = box[np.argmax(s)]
                diff = np.diff(box, axis=1)
                tr = box[np.argmin(diff)]
                bl = box[np.argmax(diff)]
                
                # 实际上你的代码用了更复杂的排序，为了稳健这里用特征点兜底
                # 如果轮廓提取成功，直接返回结果
                # (此处为了代码简洁，使用了你的原始逻辑)
                box = sorted(box, key=lambda x: x[0]) 
                left = sorted(box[:2], key=lambda x: x[1])
                right = sorted(box[2:], key=lambda x: x[1])
                src_pts = np.float32([left[0], left[1], right[1], right[0]])
                return warp_with_padding(src_pts)
    except: pass

    # 策略 2: 基于 ORB 特征点匹配 (Homography)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des_ref, des2), key=lambda x: x.distance)
            good = matches[:int(len(matches) * 0.15)]
            if len(good) >= 10:
                src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                M_homo, _ = cv2.findHomography(src_pts, dst_pts_ref, cv2.RANSAC, 5.0)
                if M_homo is not None:
                    # 注意：这里是变换到底图尺寸
                    warped = cv2.warpPerspective(img, M_homo, (w_ref, h_ref))
                    return cv2.copyMakeBorder(warped, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT)
    except: pass

    # 策略 3: 保底 (仅缩放和加黑边)
    st.warning("⚠️ 无法自动对齐，使用直接缩放模式")
    resized = cv2.resize(img, (w_ref, h_ref))
    return cv2.copyMakeBorder(resized, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT)

def calibrate_coordinates_v2(base_coords, detected_heads):
    """尝试利用识别到的线头自动微调坐标"""
    valid_offsets_x, valid_offsets_y = [], []
    for pname, (px, py) in base_coords.items():
        for head in detected_heads:
            dx, dy = head['x'] - px, head['y'] - py
            if math.sqrt(dx**2 + dy**2) < 120: 
                valid_offsets_x.append(dx); valid_offsets_y.append(dy)
    
    if not valid_offsets_x: return base_coords, False
    offset_x, offset_y = np.median(valid_offsets_x), np.median(valid_offsets_y)
    
    final_coords = {}
    for pname, (px, py) in base_coords.items():
        final_coords[pname] = [px + offset_x, py + offset_y]
    return final_coords, True

# ================= 页面主逻辑 =================
def show():
    st.markdown("## 📷 拍照判卷")
    
    # 侧边栏控制
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚙️ **识别微调**")
    conf_threshold = st.sidebar.slider("AI 自信度", 0.1, 0.9, 0.25, 0.05)
    dist_threshold = st.sidebar.slider("判定距离", 20, 100, 35, 5)
    
    st.sidebar.info("👇 如果圈圈位置整体偏移，请拖动修正")
    manual_offset_x = st.sidebar.slider("↔️ 左右平移", -200, 200, 0, 1)
    manual_offset_y = st.sidebar.slider("↕️ 上下平移", -200, 200, 0, 1)

    # 加载资源
    model, base_img, raw_pin_coords, orb_data, msg = load_resources()
    if msg != "OK": st.error(msg); return

    uploaded_file = st.file_uploader("📤 上传电路板照片", type=['jpg', 'jpeg', 'png'])
    if not uploaded_file: return

    # 1. 图像处理与对齐
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    aligned_img = align_image_robust(raw_img, base_img, orb_data)
    
    # 2. 坐标初始化 (应用 Padding)
    pin_coords = copy.deepcopy(raw_pin_coords)
    for k in pin_coords:
        pin_coords[k][0] += PADDING
        pin_coords[k][1] += PADDING

    # 3. AI 推理
    results = model(aligned_img, conf=conf_threshold, iou=0.8, verbose=False)[0]
    detected_heads = [{"color": model.names[int(b.cls[0])], "x": b.xywh[0][0].item(), "y": b.xywh[0][1].item()} for b in results.boxes]

    # 4. 坐标校准 (AI自动 + 手动)
    current_coords, is_calibrated = calibrate_coordinates_v2(pin_coords, detected_heads)
    for pin in current_coords:
        current_coords[pin][0] += manual_offset_x
        current_coords[pin][1] += manual_offset_y

    # 5. 状态判定
    board_status = {}
    for pin, (px, py) in current_coords.items():
        board_status[pin] = {"connected": False, "colors": []}
        for head in detected_heads:
            if math.sqrt((head['x']-px)**2 + (head['y']-py)**2) < dist_threshold:
                board_status[pin]["connected"] = True
                board_status[pin]["colors"].append(head['color'])

    # 6. 判卷业务逻辑 (复用你 web_app.py 的逻辑)
    err_seg, err_sig, err_pwr, err_ctrl, praises = [], [], [], [], []

    def resolve_pin_color(pin_name):
        status = board_status.get(pin_name)
        if status and status['connected']:
            for c in status['colors']: 
                if c in ['red', 'yellow', 'green', 'cyan', 'blue', 'purple', 'brown', 'black']: return c
        if pin_name in current_coords:
            px, py = current_coords[pin_name]
            return get_dominant_color(aligned_img, px, py, radius=30)
        return None

    # (此处省略部分重复的判卷 if-else，直接复用你 web_app.py 里 170行到245行的逻辑)
    # --- 简写核心逻辑以确保完整性 ---
    
    # 1. 数码管
    seg_pairs = [("U1_Pin_10 (Seg A)", "Display_Seg_A", "A段"), ("U1_Pin_12 (Seg B)", "Display_Seg_B", "B段"),
                 ("U1_Pin_13 (Seg C)", "Display_Seg_C", "C段"), ("U1_Pin_9 (Seg D)", "Display_Seg_D", "D段"),
                 ("U1_Pin_11 (Seg E)", "Display_Seg_E", "E段"), ("U1_Pin_6 (Seg F)", "Display_Seg_F", "F段"),
                 ("U1_Pin_7 (Seg G)", "Display_Seg_G", "G段")]
    seg_ok = 0
    for cp, dp, name in seg_pairs:
        c1, c2 = resolve_pin_color(cp), resolve_pin_color(dp)
        if c1: board_status[cp]['connected']=True
        if c2: board_status[dp]['connected']=True
        
        tn = cp.split(' ')[0].replace('U1_', '')
        if not c1 and not c2: err_seg.append(f"❌ **{name} 未连接**")
        elif not c1: err_seg.append(f"❌ **{name} 芯片端断路** (应接 {tn})")
        elif not c2: err_seg.append(f"❌ **{name} 数码管端断路**")
        elif c1 != c2: err_seg.append(f"❌ **{name} 颜色不匹配**(应接 {tn})")
        else: seg_ok += 1
    if seg_ok == 7: praises.append("数码管连接完美")

    # 2. 信号与电源 (CLK, VCC, GND)
    if resolve_pin_color("U1_Pin_1 (CLK)"): praises.append("时钟 CLK 已连接")
    else: err_sig.append("❌ 时钟 CLK 未连接")
    
    if resolve_pin_color("U1_Pin_16 (VCC)") == 'red': praises.append("VCC 正常")
    else: err_pwr.append("❌ VCC 供电异常 (需红线)")
    
    if resolve_pin_color("U1_Pin_8 (GND)") == 'black': praises.append("GND 正常")
    else: err_pwr.append("❌ GND 接地异常 (需黑线)")

    if resolve_pin_color("Display_COM (公共端)"): praises.append("COM端 已连接")
    else: err_pwr.append("❌ 数码管 COM 端悬空")

    # 3. 控制脚
    for pk, pn in [("U1_Pin_15 (Reset)", "复位脚"), ("U1_Pin_2 (INH)", "禁止脚")]:
        if resolve_pin_color(pk): praises.append(f"{pn} 已连接")
        else: err_ctrl.append(f"⚠️ {pn} 悬空 (建议接地)")

    # 7. 结果可视化
    col1, col2 = st.columns([1, 1])
    with col1:
        viz = aligned_img.copy()
        for p, (px, py) in current_coords.items():
            color = (0, 255, 0) if board_status[p]['connected'] else (0, 255, 255)
            cv2.circle(viz, (int(px), int(py)), dist_threshold, color, 2)
            cv2.circle(viz, (int(px), int(py)), 4, (0, 0, 255), -1)
        st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), use_column_width=True, caption=f"校准状态: {'✅' if is_calibrated else '⚠️ 需手动'}")

    with col2:
        if not is_calibrated: st.warning("⚠️ 自动对齐置信度低，请检查左侧手动平移")
        all_errs = err_seg + err_sig + err_pwr + err_ctrl
        if all_errs:
            for e in all_errs: st.error(e)
        else:
            st.success("🎉 连接逻辑完全正确！")
            st.balloons()
        with st.expander("查看检测详情", expanded=True):
            for p in praises: st.write(f"✅ {p}")