import streamlit as st
import cv2
import time
import numpy as np

def main():
    st.title("Webカメラ映像")

    st.markdown("### 1. Webカメラ映像")
    st.markdown("### 2. FPS表示付き映像")

    # 映像表示用のプレースホルダーを作成
    col1, col2 = st.columns(2)
    frame_placeholder1 = col1.empty()
    frame_placeholder2 = col2.empty()

    cap = cv2.VideoCapture(0) # 0は内蔵カメラ、複数ある場合は1,2と変える

    if not cap.isOpened():
        st.error("カメラが見つかりません。")
        return

    st.sidebar.button("停止", on_click=lambda: cap.release() and st.stop())

    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("フレームの取得に失敗しました。")
            break

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1つ目の映像 (そのまま表示)
        frame_placeholder1.image(frame_rgb, channels="RGB", use_column_width=True)

        # 2つ目の映像 (FPS表示付き)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # FPSをフレームに描画
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame_rgb_with_fps = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder2.image(frame_rgb_with_fps, channels="RGB", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()