import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# 1. 読み込み設定
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

st.set_page_config(page_title="プロ仕様ペダリング解析", layout="centered")
st.title("🚴‍♂️ ペダリング解析 & コーチング PRO")

mode = st.sidebar.radio("解析モード:", ("AIモード", "マーカーモード"))
side = st.sidebar.radio("解析脚:", ("左脚", "右脚")) if mode == "AIモード" else "マーカー"

# MediaPipe初期化
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)))

def create_report_image(max_k, a_range, fitting_adv, pedaling_adv):
    """診断レポート画像を生成する"""
    img = Image.new('RGB', (800, 800), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 30), "PEDALING & FITTING REPORT", fill=(0, 0, 0))
    d.text((50, 80), f"Max Knee Angle: {max_k} deg", fill=(0, 0, 0))
    d.text((50, 110), f"Ankle Range: {a_range} deg", fill=(0, 0, 0))
    
    d.text((50, 180), "[Fitting Advice]", fill=(255, 0, 0))
    d.text((50, 210), fitting_adv, fill=(0, 0, 0))
    
    d.text((50, 300), "[Pedaling Advice]", fill=(0, 0, 255))
    y = 330
    for line in pedaling_adv:
        d.text((50, y), f"- {line}", fill=(0, 0, 0))
        y += 30
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

uploaded_file = st.file_uploader("動画を選択 (120/240fps推奨)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    knee_angles, ankle_angles = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        results = pose.process(image)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            idx = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
                   mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX] if "左" in side else \
                  [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 
                   mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            
            p = [np.array([lm[i.value].x * w, lm[i.value].y * h]) for i in idx]
            k_angle = calculate_angle(p[0], p[1], p[2])
            a_angle = calculate_angle(p[1], p[2], p[3])
            knee_angles.append(k_angle)
            ankle_angles.append(a_angle)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        st_frame.image(image, channels="RGB", use_container_width=True)
    cap.release()

    if knee_angles:
        st.divider()
        max_k = int(max(knee_angles))
        a_max, a_min = max(ankle_angles), min(ankle_angles)
        a_range = int(a_max - a_min)
        
        # 1. フィッティング・アドバイス
        if max_k < 145: fitting_adv = "Saddle too LOW. Raise 10-20mm."
        elif 145 <= max_k <= 155: fitting_adv = "Saddle height is PERFECT."
        else: fitting_adv = "Saddle too HIGH. Lower 5-10mm."

        # 2. ペダリング・アドバイス
        pedaling_adv = []
        if a_range > 20:
            pedaling_adv.append("Excessive Ankling: Stabilize your ankle joint.")
            pedaling_adv.append("Try to push with your whole leg, not just the foot.")
        else:
            pedaling_adv.append("Stable Ankle: Good power transfer.")
        
        if a_min < 40:
            pedaling_adv.append("Heel Drop: Avoid dropping the heel too much at 3 o'clock.")
        
        pedaling_adv.append("Focus on pulling up from 7 to 10 o'clock.")

        # レポート表示
        st.subheader("💡 競技者向け総合アドバイス")
        st.write(f"**【車体調整】**: {fitting_adv}")
        st.write("**【ペダリング技術】**:")
        for adv in pedaling_adv:
            st.write(f"- {adv}")

        # 画像保存ボタン
        report_bytes = create_report_image(max_k, a_range, fitting_adv, pedaling_adv)
        st.download_button("📋 診断レポート画像を保存", report_bytes, "report.png", "image/png")
        
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
