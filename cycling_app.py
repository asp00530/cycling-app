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

st.set_page_config(page_title="ペダリング解析PRO", layout="centered")
st.title("🚴‍♂️ ペダリング・アナライザー PRO")

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

def create_report_image(max_k, a_range, advice_list):
    """診断レポート画像を生成する"""
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        # フォント設定（標準的なものを使用）
        f_title = ImageFont.load_default()
    except:
        f_title = ImageFont.load_default()

    d.text((50, 30), "PEDALING ANALYSIS REPORT", fill=(0, 0, 0))
    d.text((50, 100), f"Max Knee Angle: {max_k} deg", fill=(0, 0, 0))
    d.text((50, 150), f"Ankle Range: {a_range} deg", fill=(0, 0, 0))
    
    y = 250
    d.text((50, y-30), "[Advice]", fill=(0, 0, 0))
    for line in advice_list:
        d.text((50, y), f"- {line}", fill=(0, 0, 0))
        y += 40
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

uploaded_file = st.file_uploader("動画を選択", type=["mp4", "mov", "avi"])

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
            cv2.putText(image, f"Knee: {int(k_angle)}deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        st_frame.image(image, channels="RGB", use_container_width=True)
    
    cap.release()

    if knee_angles:
        st.divider()
        max_k = int(max(knee_angles))
        a_range = int(max(ankle_angles) - min(ankle_angles)) if ankle_angles else 0
        
        # アドバイス生成
        advices = []
        if max_k < 145: advices.append(f"Saddle is LOW ({max_k} deg). Raise 10-20mm.")
        elif 145 <= max_k <= 155: advices.append("Saddle height is PERFECT.")
        else: advices.append(f"Saddle is HIGH ({max_k} deg). Lower 5-10mm.")
        
        if a_range > 20: advices.append(f"Ankling is LARGE ({a_range} deg). Fix your ankle.")
        else: advices.append("Ankle stability is GOOD.")

        # レポート画像作成ボタン
        report_bytes = create_report_image(max_k, a_range, advices)
        st.download_button(
            label="📋 診断レポート画像を保存",
            data=report_bytes,
            file_name="pedaling_report.png",
            mime="image/png"
        )
        
        st.line_chart({"膝角度": knee_angles, "足首角度": ankle_angles})
        st.write(f"最大膝角度: {max_k}° / 足首可動幅: {a_range}°")
