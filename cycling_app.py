import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

# 1. ページの設定
st.set_page_config(page_title="ハイブリッド解析PRO", layout="centered")
st.title("🚴‍♂️ ペダリング・アナライザー Extreme")

# サイドバー設定
st.sidebar.header("解析設定")
mode = st.sidebar.radio("解析モード:", ("AIモード (マーカーなし)", "マーカーモード (色追跡)"))

if mode == "マーカーモード (色追跡)":
    # メッセージを修正しました
    st.sidebar.info("腰・膝・足首・つま先の4点に同じ色のシールを貼ってください。")
    h_range = st.sidebar.slider("色相(H)範囲", 0, 180, (140, 175)) 
    s_min = st.sidebar.slider("彩度(S)最小", 0, 255, 100)
else:
    side = st.sidebar.radio("解析する脚:", ("左脚 (Left)", "右脚 (Right)"))

# 2. 初期設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 3. 動画アップロード
uploaded_file = st.file_uploader("動画を選択 (120/240fps推奨)", type=["mp4", "mov", "avi"])

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
                IDS = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
                       mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX] if "左" in side else \
                      [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, 
                       mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                
                p = [np.array([lm[i.value].x * w, lm[i.value].y * h]) for i in IDS]
                k_angle = calculate_angle(p[0], p[1], p[2])
                a_angle = calculate_angle(p[1], p[2], p[3])
                if 85 < k_angle < 95: knee_x_offsets.append(p[1][0] - p[2][0])
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else: # マーカーモード
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([h_range[0], s_min, 50])
            upper = np.array([h_range[1], 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                M = cv2.moments(c)
                if M["m00"] > 0: centers.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
            
            if len(centers) >= 3:
                # Y座標(高さ)でソート
                centers = sorted(centers, key=lambda x: x[1]) 
                hip = centers[0]
                knee = centers[1]
                ankle = centers[2]
                
                k_angle = calculate_angle(hip, knee, ankle)
                if len(centers) == 4:
                    toe = centers[3] # 一番低い（または一番前）を「つま先」とする
                    a_angle = calculate_angle(knee, ankle, toe)
                
                if 85 < k_angle < 95: knee_x_offsets.append(knee[0] - ankle[0])
                for pt in centers: cv2.circle(image, tuple(pt), 10, (255, 255, 255), -1)

        if k_angle:
            knee_angles.append(k_angle)
            if a_angle: ankle_angles.append(a_angle)
            cv2.putText(image, f"Knee: {int(k_angle)}deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        st_frame.image(image, channels="RGB", use_container_width=True)
    
    cap.release()

    # 4. 総合診断レポート
    if knee_angles:
        st.divider()
        st.subheader("📊 解析レポート")
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
        
        max_k = int(max(knee_angles))
        a_min, a_max = (min(ankle_angles), max(ankle_angles)) if ankle_angles else (0, 0)
        a_range = int(a_max - a_min)
        
        col1, col2 = st.columns(2)
        col1.metric("最大膝角度", f"{max_k}°")
        col2.metric("足首可動幅", f"{a_range}°")

        st.subheader("💡 競技者向け総合アドバイス")
        # 高さ診断
        if max_k < 145: st.error(f"【高さ】低すぎます({max_k}°)。サドルを15mm程度上げてください。")
        elif 145 <= max_k <= 155: st.success(f"【高さ】理想的です。")
        else: st.error(f"【高さ】高すぎます({max_k}°)。5-10mm下げてください。")

        # 前後位置診断
        if knee_x_offsets:
            off = np.mean(knee_x_offsets)
            # 画像サイズに依存するため、相対的な値で判定
            if off > 20: st.error("【前後位置】サドルが前すぎます。後ろへ引いてください。")
            elif off < -20: st.warning("【前後位置】サドルが後ろすぎます。少し前に出してください。")
            else: st.success("【前後位置】適正です。膝とペダル軸が揃っています。")

        # 足首診断
        if a_range > 25: st.error(f"【足首】アンクリングが極端に大きいです({a_range}°)。サドル高の不適合か、踏み込みの癖を修正しましょう。")
        elif 15 < a_range <= 25: st.warning(f"【足首】やや動きが大きいです。足首の固定を意識してください。")
        else: st.success("【足首】非常に安定した綺麗なペダリングです。")