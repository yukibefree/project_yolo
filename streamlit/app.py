import streamlit as st
import cv2
import time
import numpy as np
import os
import sys

# プロジェクトルートの設定
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from tracker.tracker import Tracker

def main():
    st.title("Webカメラ映像")

    # 映像表示用のプレースホルダーを作成
    frame_placeholder1 = st.empty()
    frame_placeholder2 = st.empty()

    tracker = Tracker()

    if not tracker.cap.isOpened():
        st.error("カメラが見つかりません。")
        return

    prev_time = time.time()
    while tracker.cap.isOpened():
        ret, frame = tracker.cap.read()
        if not ret:
            st.warning("フレームの取得に失敗しました。")
            break

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1つ目の映像 (そのまま表示)
        frame_placeholder1.image(frame_rgb, channels="RGB", width='content')

        # 2つ目の映像 (FPS表示付き)
        frame_processed = tracker.track(frame, server=False)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # FPSをフレームに描画
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame_processed, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame_rgb_with_fps = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        frame_placeholder2.image(frame_rgb_with_fps, channels="RGB", width='content')

    tracker.close_camera()

if __name__ == "__main__":
    main()