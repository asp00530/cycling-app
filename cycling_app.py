import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# 1. 読み込み設定（エラー回避型）
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

# ページ基本設定
st.set_page_config(page_title="プロ仕様ペダリング解析", layout="centered")
st.title("🚴‍♂️ ペダリング解析 & コーチング PRO")

# サイドバー：設定
st.sidebar.header("解析モード設定")
mode = st.sidebar.radio("解析モード:", ("AIモード (シールなし)", "マーカーモード (シールあり)"))

if mode == "マーカーモード (シールあり)":
    st.sidebar.info("腰・膝・足首・つま先の4点に同じ色のシールを貼ってください。")
    h_range = st.sidebar.slider("色相(H)の範囲", 0, 180, (140, 175), help="シールの色が白く浮かび上がるように調整してください") 
    s_min = st.sidebar.slider("鮮やかさ(S)最小", 0, 255, 70)
    v_min = st.sidebar.slider("明るさ(V)最小", 0, 255, 50)
else:
    side = st.sidebar.radio("解析する脚:", ("左脚 (Left)", "右脚 (Right)"))

# AI初期化
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)))

def create_report_image(max_k, a_range, fitting_adv, pedaling_adv):
    img = Image.new('RGB', (800, 800), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 30), "PEDALING & FITTING REPORT", fill=(0, 0, 0))
    d.text((50, 80), f"Max Knee Angle: {max_k} deg", fill=(0, 0, 0))
    d.text((50, 110), f"Ankle Range: {a_range} deg", fill=(0, 0, 0))
    d.text((50, 180), f"[Fitting]: {fitting_adv}", fill=(255, 0, 0))
    y = 250
    for line in pedaling_adv:
        d.text((50, y), f"- {line}", fill=(0, 0, 255))
        y += 40
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
        k_angle, a_angle = 0, 0

        if mode == "AIモード (シールなし)":
            results = pose.process(image)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                IDS = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
                       mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX] if "左" in side else \
                      [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 
                       mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                p = [np.array([lm[i.value].x * w, lm[i.value].y * h]) for i in IDS]
                k_angle = calculate_angle(p[0], p[1], p[2])
                a_angle = calculate_angle(p[1], p[2], p[3])
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            # マーカー認識強化ロジック
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([h_range[0], s_min, v_min]), np.array([h_range[1], 255, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                if cv2.contourArea(c) > 50:
                    M = cv2.moments(c)
                    if M["m00"] > 0: centers.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            if len(centers) >= 3:
                centers = sorted(centers, key=lambda x: x[1]) # 高さでソート
                k_angle = calculate_angle(centers[0], centers[1], centers[2])
                if len(centers) == 4: a_angle = calculate_angle(centers[1], centers[2], centers[3])
                for pt in centers: cv2.circle(image, tuple(pt), 10, (255, 255, 255), -1)

        if k_angle > 0:
            knee_angles.append(k_angle)
            if a_angle > 0: ankle_angles.append(a_angle)
            cv2.putText(image, f"Knee: {int(k_angle)}deg", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        st_frame.image(image, channels="RGB", use_container_width=True)
    
    cap.release()

    if knee_angles:
        st.divider()
        max_k = int(max(knee_angles))
        a_range = int(max(ankle_angles) - min(ankle_angles)) if ankle_angles else 0
        
        # 診断とアドバイス
        fitting_adv = "理想的です" if 145 <= max_k <= 155 else ("サドルを上げて" if max_k < 145 else "サドルを下げて")
        pedaling_adv = ["足首を固定しましょう" if a_range > 20 else "足首の安定感GOOD", "引き足を意識しましょう"]

        st.subheader("💡 診断結果")
        st.write(f"**車体調整**: {fitting_adv} (最大膝角度: {max_k}°)")
        st.write("**スキル**: " + " / ".join(pedaling_adv))

        # 画像保存ボタン
        report_bytes = create_report_image(max_k, a_range, fitting_adv, pedaling_adv)
        st.download_button("📋 診断レポート画像を保存", report_bytes, "report.png", "image/png")
        
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
