import cv2
import os
import sys
from ultralytics import solutions
import asyncio

# プロジェクトファイルパスを通す
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(os.path.join(PROJECT_ROOT))

# ソースファイルパス
SOURCE = os.path.join(PROJECT_ROOT, "solutions", "parking", "source")
VIDEO_PATH = os.path.join(SOURCE, "parking_aso.mp4")
MODEL_FILE = os.path.join(PROJECT_ROOT, "models", "yolo11n.pt")
JSON_FILE = os.path.join(SOURCE, "bounding_boxes.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "parking_management.mp4")

class ParkingManagement:
    def __init__(self, model=MODEL_FILE, json_file=JSON_FILE):
        self.model = model
        self.json_file = json_file
        self.setup_video(VIDEO_PATH)
        self.setup_video_writer()

        self.parkingmanager = solutions.ParkingManagement(
            model=self.model,  # path to model file
            json_file=self.json_file,  # path to parking annotations file
        )

    def setup_video(self, path: str):
        """動画のセットアップをする。"""
        self.cap = cv2.VideoCapture(path)
        assert self.cap.isOpened(), "Error reading video file"

    def setup_video_writer(self):
        """
        ビデオライターをセットアップします。
        """
        # Initialize video writer parameters
        w, h, fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def exec(self):
        while self.cap.isOpened():
            ret, im0 = self.cap.read()
            if not ret:
                break

            results = self.parkingmanager(im0)

            self.video_writer.write(results.plot_im)  # write the processed frame.
            # 画像（NumPy配列）をJPEGに変換
            ret, jpeg = cv2.imencode('.jpg', results.plot_im)
            if not ret:
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )

        self.release()

    async def ws_exec(self):
        while self.cap.isOpened():
            ret, im0 = self.cap.read()
            if not ret:
                break

            results = self.parkingmanager(im0)

            self.video_writer.write(results.plot_im)  # write the processed frame.

            # 画像（NumPy配列）をJPEGに変換
            ret, jpeg = cv2.imencode('.jpg', results.plot_im)
            if not ret:
                continue
            yield jpeg.tobytes()

            await asyncio.sleep(0.01)

        self.release()
    
    def stream_exec(self):
        while self.cap.isOpened():
            ret, im0 = self.cap.read()
            if not ret:
                break

            results = self.parkingmanager(im0)

            self.video_writer.write(results.plot_im)  # write the processed frame.
            # 画像（NumPy配列）をRGBに変換
            frame_rgb = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2RGB)
            yield frame_rgb
        self.release()

    def release(self):
        """リソースを解放する。"""
        self.cap.release()
        self.video_writer.release()

if __name__ == '__main__':
    parkingmanager = ParkingManagement()
    parkingmanager.exec()
    sys.exit(0)