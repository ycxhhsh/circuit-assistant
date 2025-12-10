# --- filename: page_recognition.py ---
import os
import sys

# 🔥 1. 核心修复：配置目录重定向 (必须在最开头)
# 解决 "user config directory is not writable" 警告
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import copy
import gc

# ================= 1. 资源加载 =================
@st.cache_resource
def load_resources():
    try:
        model = YOLO('best.pt') 
        base_img = cv2.imread('base_fixed.jpg')
        if base_img is None: return None, None, None, None, "❌ 找不到 base_fixed.jpg"
        
        with open('board_config.json', 'r', encoding='utf-8') as f:
            pin_coords = json.load(f)
            
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), None)
        
        return model, base_img, pin_coords, (sift, kp_ref, des_ref), "OK"
    except Exception as e:
        return None, None, None, None, str(e)

# ================= 2. 图像处理与压缩 =================
PADDING = 40 

# 🔥 2. 核心修复：图片压缩函数
# 防止 4000px 大图直接塞进内存导致 "Oh no" 崩溃
def resize_if_too_large(img, max_width=1024):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        return cv2.resize(img, (max_width, new_h))
    return img

def correct_orientation(img):
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def align_image_sift(raw_img, base_img, feature_data):
    sift, kp_ref, des_ref = feature_data
    h_ref, w_ref = base_img.shape[:2]
    img = correct_orientation(raw_img)
    w_new, h_new = w_ref + 2 * PADDING, h_ref + 2 * PADDING

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_img, des_img = sift.detectAndCompute(gray, None)

        if des_img is not None and len(kp_img) > 10:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_img, des_ref, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > 10:
                src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    T = np.array([[1, 0, PADDING], [0, 1, PADDING], [0, 0, 1]])
                    M_final = T.dot(M)
                    warped = cv2.warpPerspective(img, M_final, (w_new, h_new))
                    return warped
    except Exception as e:
        pass 

    resized = cv2.resize(img, (w_ref, h_ref))
    return cv2.copyMakeBorder(resized, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT)

def calibrate_coordinates(base_coords, detected_heads):
    valid_offsets_x, valid_offsets_y = [], []
    for pname, (px, py) in base_coords.items():
        min_dist = 9999
        nearest_head = None
        for head in detected_heads:
            d = math.sqrt((head['x'] - px)**2 + (head['y'] - py)**2)
            if d < min_dist: min_dist = d; nearest_head = head
        if min_dist < 60: 
            valid_offsets_x.append(nearest_head['x'] - px)
            valid_offsets_y.append(nearest_head['y'] - py)
    if not valid_offsets_x: return base_coords, False
    offset_x, offset_y = np.median(valid_offsets_x), np.median(valid_offsets_y)
    final_coords = {}
    for pname, (px, py) in base_coords.items():
        final_coords[pname] = [px + offset_x, py + offset_y]
    return final_coords, True

# ================= 3. 页面主逻辑 =================
def show():
    st.markdown("## 📷 AI 智能电路辅助判卷系统")
    
    st.sidebar.markdown("---")
    conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.05, 0.9, 0.15, 0.05) 
    dist_threshold = st.sidebar.slider("欧氏距离判定范围 (px)", 20, 150, 35, 5) 
    
    st.sidebar.info("工程参数校准")
    manual_offset_x = st.sidebar.slider("X 轴偏移校正", -100, 100, 0)
    manual_offset_y = st.sidebar.slider("Y 轴偏移校正", -100, 100, 0)

    model, base_img, raw_pin_coords, feature_data, msg = load_resources()
    if msg != "OK": st.error(msg); return

    uploaded_file = st.file_uploader("上传待检测电路图像", type=['jpg', 'jpeg', 'png'])
    if not uploaded_file: return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    if raw_img is None:
        st.error("无法解析图片")
        return

    # 🔥 3. 核心修复：调用压缩
    # 在进行任何 AI 处理前，先把图片压到 1024px 宽，救命的一步！
    process_img = resize_if_too_large(raw_img, max_width=1024)
    gc.collect() # 主动释放内存

    aligned_img = align_image_sift(process_img, base_img, feature_data)
    
    pin_coords = copy.deepcopy(raw_pin_coords)
    for k in pin_coords: pin_coords[k][0] += PADDING; pin_coords[k][1] += PADDING

    results = model(aligned_img, conf=conf_threshold, verbose=False)[0]
    detected_heads = [{"color": model.names[int(b.cls[0])], "x": b.xywh[0][0].item(), "y": b.xywh[0][1].item()} for b in results.boxes]

    current_coords, is_calibrated = calibrate_coordinates(pin_coords, detected_heads)
    for pin in current_coords:
        current_coords[pin][0] += manual_offset_x
        current_coords[pin][1] += manual_offset_y

    viz_img = aligned_img.copy()

    # === 🔥 1. 绘图层：强制画线逻辑 (严格保留您要求的版本) ===
    
    # 1.1 画所有引脚的“扫描圈” (绿色空心圆)
    for pname, (px, py) in current_coords.items():
        cv2.circle(viz_img, (int(px), int(py)), 12, (0, 255, 0), 2) 

    # 1.2 定义任务 (Pin 1, 2, 3, 15)
    tasks = [
        {
            "name": "Pin 1 连接时钟 (CLK)", 
            "pin": "U1_Pin_1 (CLK)", "dest": "Button_CLK", 
            "color_cn": "橙色", "wire_color": (0, 165, 255), "expect_cls": "head_orange"
        },
        {
            "name": "Pin 2 连接接地 (INH)", 
            "pin": "U1_Pin_2 (INH)", "dest": "GND_Input", 
            "color_cn": "紫色", "wire_color": (255, 0, 255), "expect_cls": "head_purple"
        },
        {
            "name": "Pin 3 连接电源 (VCC)", 
            "pin": "U1_Pin_3 (DE1)", "dest": "U1_Pin_16 (VCC)", 
            "color_cn": "蓝色", "wire_color": (255, 200, 0), "expect_cls": "head_blue"
        },
        {
            "name": "Pin 15 复位接地 (RST)", 
            "pin": "U1_Pin_15 (Reset)", "dest": "GND_Screw", 
            "color_cn": "白色", "wire_color": (200, 200, 200), "expect_cls": "head_white"
        }
    ]

    # 1.3 强制绘线 (Pre-draw): 直接用理论坐标把线画出来
    for task in tasks:
        if task['pin'] in current_coords and task['dest'] in current_coords:
            pt1 = current_coords[task['pin']]
            pt2 = current_coords[task['dest']]
            
            p1_int = (int(pt1[0]), int(pt1[1]))
            p2_int = (int(pt2[0]), int(pt2[1]))
            
            # 画实心端点
            cv2.circle(viz_img, p1_int, 6, task['wire_color'], -1)
            cv2.circle(viz_img, p2_int, 6, task['wire_color'], -1)

    # === 2. 逻辑检测层 (仅用于更新UI文字) ===
    
# === 2. 逻辑检测层 (更新后的动态互补逻辑) ===
    
    # 辅助函数：支持动态阈值检测 (替换原来的 check_point_loose)
    def check_point_dynamic(coord_key, target_cls, dynamic_threshold):
        if coord_key not in current_coords: return False
        px, py = current_coords[coord_key]
        for head in detected_heads:
            # 1. 颜色匹配
            if target_cls in head['color']:
                # 2. 距离匹配
                dist = math.sqrt((head['x'] - px)**2 + (head['y'] - py)**2)
                
                # 3. 动态阈值判定
                # 如果是“宽容模式”，我们允许检测到的点偏离得更远一点
                if dist < dist_threshold + dynamic_threshold: 
                    return True
        return False

    cols = st.columns(2)
    with cols[1]:
        st.write("#### 🛡️ 逻辑连接检测 (双端一致性校验)")
        for task in tasks:
            # --- 核心策略：动态阈值互补 ---
            
            # 第一轮：用正常标准看两头 (0增益)
            p1_strict = check_point_dynamic(task['pin'], task['expect_cls'], 0)
            p2_strict = check_point_dynamic(task['dest'], task['expect_cls'], 0)

            final_status = False
            
            # 情况A：两头都很完美 -> 完美通过
            if p1_strict and p2_strict:
                final_status = True
            
            # 情况B：只有一头很完美 -> 触发“视觉补偿机制”
            # 既然一头已经连上了，我们把另一头的判定范围扩大 (放宽 60px) 再找一次
            elif p1_strict:
                p2_loose = check_point_dynamic(task['dest'], task['expect_cls'], 60)
                if p2_loose: final_status = True
                
            elif p2_strict:
                p1_loose = check_point_dynamic(task['pin'], task['expect_cls'], 60)
                if p1_loose: final_status = True

            # --- 结果展示 ---
            if final_status:
                st.markdown(f"✅ **{task['name']}**: 双端信号闭环 ({task['color_cn']}线)")
            else:
                # 即使失败，如果有一头识别到了，给个黄色警告而不是红色错误，演示效果更好
                if p1_strict or p2_strict:
                     st.markdown(f"⚠️ **{task['name']}**: 信号单端接入，请检查另一端 ({task['color_cn']}线)")
                else:
                     st.markdown(f"❌ **{task['name']}**: 未检测到信号链路")
