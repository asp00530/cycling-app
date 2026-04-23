import streamlit as st
import cv2
import numpy as np
import tempfile

# ここが重要：mediapipe.solutions から直接インポートします
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

# 1. ページの設定
st.set_page_config(page_title="ハイブリッド解析PRO", layout="centered")
st.title("🚴‍♂️ ペダリング・アナライザー Extreme")

# サイドバー設定
st.sidebar.header("解析設定")
mode = st.sidebar.radio("解析モード:", ("AIモード (マーカーなし)", "マーカーモード (色追跡)"))

if mode == "マーカーモード (色追跡)":
    st.sidebar.info("腰・膝・足首・つま先の4点に同じ色のシールを貼ってください。")
    h_range = st.sidebar.slider("色相(H)範囲", 0, 180, (140, 175)) 
    s_min = st.sidebar.slider("彩度(S)最小", 0, 255, 100)
else:
    side = st.sidebar.radio("解析する脚:", ("左脚 (Left)", "右脚 (Right)"))

# 2. 初期設定
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 3. 動画アップロード
uploaded_file = st.file_uploader("動画を選択", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    
    knee_angles, ankle_angles, knee_x_offsets = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        k_angle, a_angle = None, None

        if mode == "AIモード (マーカーなし)":
            results = pose.process(image)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # 左右の判定
                ID_LIST = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
                           mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX] if "左" in side else \
                          [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 
                           mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                
                p = [np.array([lm[i.value].x * w, lm[i.value].y * h]) for i in ID_LIST]
                k_angle = calculate_angle(p[0], p[1], p[2])
                a_angle = calculate_angle(p[1], p[2], p[3])
                if 85 < k_angle < 95: knee_x_offsets.append(p[1][0] - p[2][0])
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else: # マーカーモード
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([h_range[0], s_min, 50]), np.array([h_range[1], 255, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                M = cv2.moments(c)
                if M["m00"] > 0: centers.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            
            if len(centers) >= 3:
                centers = sorted(centers, key=lambda x: x[1]) # Y座標でソート
                k_angle = calculate_angle(centers[0], centers[1], centers[2])
                if len(centers) == 4:
                    a_angle = calculate_angle(centers[1], centers[2], centers[3])
                if 85 < k_angle < 95: knee_x_offsets.append(centers[1][0] - centers[2][0])
                for pt in centers: cv2.circle(image, tuple(pt), 10, (255, 255, 255), -1)

        if k_angle:
            knee_angles.append(k_angle)
            if a_angle: ankle_angles.append(a_angle)
            cv2.putText(image, f"Knee: {int(k_angle)}deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        st_frame.image(image, channels="RGB", use_container_width=True)
    
    cap.release()

    if knee_angles:
        st.divider()
        st.subheader("📊 解析レポート")
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
        max_k = int(max(knee_angles))
        st.metric("最大膝角度", f"{max_k}°")
        
        # 診断アドバイス
        if max_k < 145: st.error("サドルが低すぎます。10mm以上上げてください。")
        elif 145 <= max_k <= 155: st.success("サドル高は理想的です。")
        else: st.error("サドルが高すぎます。5mmほど下げてください。")
