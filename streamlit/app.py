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
from solutions.parking.parking_management import ParkingManagement

# 動画像設定
url = 'https://www.youtube.com/watch?v=XBnob1Mps4g'

def exec_streamlit():
    # ページ設定
    st.set_page_config(layout="wide")
    st.title("Webカメラ映像")

    # 横並びに2つのカラムを作成
    col1, col2 = st.columns(2)

    # 映像表示用のプレースホルダーを作成
    with col1:
        frame_placeholder1 = st.empty()
    with col2:
        frame_placeholder2 = st.empty()

    tracker = Tracker()

    if not tracker.cap.isOpened():
        st.error("カメラが見つかりません。")
        return

    prev_time = time.time()
    while tracker.cap.isOpened():
        if not tracker.url:
          # カメラからフレームを取得
          ret, frame = tracker.cap.read()
          if not ret:
            st.warning("フレームの取得に失敗しました。")
            break
        else:
          # YouTube動画からフレームを取得
          _, images, _ = next(tracker.load_streams)
          if images is None:
              st.warning("フレームの取得に失敗しました。")
              break
          else:
              frame = images[0]

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1つ目の映像 (そのまま表示)
        frame_placeholder1.image(frame_rgb, channels="RGB", width='content')

        # 2つ目の映像 (FPS表示付き)
        frame_processed = tracker.track(frame)

        frame_rgb_with_fps = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
        frame_placeholder2.image(frame_rgb_with_fps, channels="RGB", width='content')

    tracker.release_capture()

def exec_parking_streamlit():
    # ページ設定
    st.set_page_config(layout="wide")
    st.title("駐車場管理映像")

    # 横並びに2つのカラムを作成
    col1, col2 = st.columns(2)

    # 映像表示用のプレースホルダーを作成
    with col1:
        frame_placeholder1 = st.empty()

    parking_manager = ParkingManagement()
    ws_parking_manager = ParkingManagement()

    while parking_manager.cap.isOpened() or ws_parking_manager.cap.isOpened():
        ret, im0 = parking_manager.cap.read()
        ws_ret, ws_im0 = ws_parking_manager.cap.read()
        if not ret and not ws_ret:
            break

        results = parking_manager.parkingmanager(im0)
        ws_results = ws_parking_manager.parkingmanager(ws_im0)

        parking_manager.video_writer.write(results.plot_im)  # write the processed frame.
        ws_parking_manager.video_writer.write(ws_results.plot_im)  # write the processed frame.
        
        # 画像（NumPy配列）をRGBに変換
        frame_rgb = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2RGB)
        frame_placeholder1.image(frame_rgb, channels="RGB", width='content')

    parking_manager.release()


if __name__ == "__main__":
    #exec_streamlit()
    exec_parking_streamlit()