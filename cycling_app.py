import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageDraw
import io

# 1. ライブラリ読み込み
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

st.set_page_config(page_title="プロ仕様ペダリング解析", layout="centered")
st.title("🚴‍♂️ ペダリング解析 & 精密コーチング PRO")

# サイドバー設定
st.sidebar.header("解析モード設定")
mode = st.sidebar.radio("解析モード:", ("AIモード (シールなし)", "マーカーモード (シールあり)"))

if mode == "マーカーモード (シールあり)":
    st.sidebar.info("腰・膝・足首・つま先の4点に同色のシールを貼ってください。")
    h_range = st.sidebar.slider("色相(H)の範囲", 0, 180, (140, 175)) 
    s_min = st.sidebar.slider("鮮やかさ(S)最小", 0, 255, 70)
    v_min = st.sidebar.slider("明るさ(V)最小", 0, 255, 50)
else:
    side = st.sidebar.radio("解析する脚:", ("左脚", "右脚"))

# AI初期化
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0: return 0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)))

def create_report_image(max_k, a_range, f_title, f_detail, p_advs):
    img = Image.new('RGB', (800, 900), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 30), "PEDALING & FITTING REPORT", fill=(0, 0, 0))
    d.text((50, 80), f"Max Knee Angle: {max_k} deg / Ankle Range: {a_range} deg", fill=(0, 0, 0))
    d.text((50, 150), f"[Fitting Advice]", fill=(255, 0, 0))
    d.text((50, 180), f"{f_title}: {f_detail}", fill=(0, 0, 0))
    d.text((50, 280), "[Pedaling Technique Advice]", fill=(0, 0, 255))
    y = 320
    for line in p_advs:
        d.text((50, y), f"- {line}", fill=(0, 0, 0))
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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([h_range[0], s_min, v_min]), np.array([h_range[1], 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                if cv2.contourArea(c) > 30:
                    M = cv2.moments(c)
                    if M["m00"] > 0: centers.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            if len(centers) >= 3:
                centers = sorted(centers, key=lambda x: x[1]) 
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
        
        st.subheader("📊 競技者向け精密診断レポート")
        st.markdown("### 🛠️ フィッティング・アドバイス")
        if max_k < 145:
            f_t, f_d = "サドルが低い", f"現在 {max_k}°。145°-155°が理想。10-20mm上げを推奨。"
            st.error(f"**{f_t}**: {f_d}")
        elif 145 <= max_k <= 155:
            f_t, f_d = "適正", "理想的です。維持してください。"
            st.success(f"**{f_t}**: {f_d}")
        else:
            f_t, f_d = "サドルが高い", f"現在 {max_k}°。膝裏の故障リスク。5-10mm下げを推奨。"
            st.error(f"**{f_t}**: {f_d}")

        st.markdown("### 🦵 ペダリング技術アドバイス")
        p_advs = []
        if a_range > 20:
            p_advs.append(f"足首のブレ({a_range}°)大。固定を意識しましょう。")
        else:
            p_advs.append("足首が安定した綺麗なペダリングです。")
        p_advs.append("6時位置で「靴裏の泥を払う」ように後ろへ引く意識を。")
        for adv in p_advs: st.info(adv)

        rb = create_report_image(max_k, a_range, f_t, f_d, p_advs)
        st.download_button("📋 診断レポート画像を保存", rb, "pedaling_report.png", "image/png")
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
