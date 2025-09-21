# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import time
import os
import sys
import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

# プロジェクトルートの設定
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.select_camera import SelectCamera

class Tracker:
    def __init__(self):
        """設定を初期化します。"""

        # --- 環境設定 ---
        # GPU (CUDA) を使用するかどうか。Trueにすると、処理速度が大幅に向上します。
        enable_gpu = True
        # 実行環境がMac PCであるかを設定します。
        mac_pc = True

        # --- モデル設定 ---
        # モデルファイル（.pt）が保存されているディレクトリのパス
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        # 使用するYOLOv8モデルの名前。'n'は「nano」を表す軽量で高速なモデルです。
        model_name = "yolo12n.pt"
        # モデルファイルへの完全なパス
        model_file = os.path.join(model_path, model_name)

        # --- 動画・表示設定 ---
        # 画面左上に現在のFPS（1秒あたりのフレーム数）を表示するかどうか。
        self.show_fps = True
        # 検出されたオブジェクトの信頼度（Confidence Score）を表示するかどうか。
        self.show_conf = False
        # 処理後の映像を動画ファイルとして保存するかどうか。
        self.save_video = False
        # 出力動画を保存するディレクトリのパス
        video_path = os.path.join(root_dir, 'output')
        # 保存する動画のファイル名
        video_name = "interactive_tracker_output.avi"
        # 出力動画ファイルへの完全なパス
        self.video_output_path = os.path.join(video_path, video_name)

        # --- トラッキング対象とYOLOモデル設定 ---
        # 追跡するオブジェクトの種類をクラスIDで指定します。
        # COCOデータセットのID: 0は人(person)、2は車(car)です。
        PERSON_CLASS_ID = 0
        CAR_CLASS_ID = 2
        self.target_classes = [PERSON_CLASS_ID, CAR_CLASS_ID]

        # オブジェクトとして認識する最低限の信頼度。低いと誤検出が増える可能性があります。
        self.conf = 0.3
        # IoU (Intersection over Union) の閾値。この値が低いと、重なり合うボックスがより多く許容されます。
        self.iou = 0.3
        # 1つの画像で検出するオブジェクトの最大数。
        self.max_det = 20

        # --- トラッカー設定 ---
        # 使用するトラッカーの種類を指定
        self.tracker = "bytetrack.yaml"
        # トラッカーの動作をカスタマイズするための引数。
        self.track_args = {
            # フレーム履歴をストリームとして保持し、継続的な追跡を可能にします。
            "persist": True,
            # トラッカーのデバッグ情報をコンソールに表示するかどうか。
            "verbose": False,
        }

        # リアルタイム映像が表示されるウィンドウのタイトル。
        self.window_name = "Ultralytics YOLO Interactive Tracking"

        LOGGER.info("🚀 Initializing model...")
        if enable_gpu:
            LOGGER.info("Using GPU...")
            self.model = YOLO(model_file)
            if mac_pc:
                self.model.to("mps")
            else:
                self.model.to("cuda")
        else:
            LOGGER.info("Using CPU...")
            self.model = YOLO(model_file, task="detect")

        self.classes = self.model.names  # Store model class names
        self.setup_camera()
        self.setup_video_writer()

    def setup_camera(self):
        """
        カメラをセットアップします。
        """
        # カメラの選択
        select_camera = SelectCamera()
        camera_index = select_camera.get_camera_index()
        self.cap = cv2.VideoCapture(camera_index)

    def setup_video_writer(self):
        """
        ビデオライターをセットアップします。
        """
        if not self.cap.isOpened():
            LOGGER.error("カメラが開かれていません。ビデオライターをセットアップできません。")
            return
        # Initialize video writer
        self.vw = None
        if self.save_video:
            self.w, self.h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            self.vw = cv2.VideoWriter(self.video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))

        self.selected_object_id = None
        selected_bbox = None
        selected_center = None


    def get_center(self, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
        """
        Calculate the center point of a bounding box.

        Args:
            x1 (int): Top-left X coordinate.
            y1 (int): Top-left Y coordinate.
            x2 (int): Bottom-right X coordinate.
            y2 (int): Bottom-right Y coordinate.

        Returns:
            center_x (int): X-coordinate of the center point.
            center_y (int): Y-coordinate of the center point.
        """
        return (x1 + x2) // 2, (y1 + y2) // 2


    def extend_line_from_edge(self, mid_x: int, mid_y: int, direction: str, img_shape: tuple[int, int, int]) -> tuple[int, int]:
        """
        Calculate the endpoint to extend a line from the center toward an image edge.

        Args:
            mid_x (int): X-coordinate of the midpoint.
            mid_y (int): Y-coordinate of the midpoint.
            direction (str): Direction to extend ('left', 'right', 'up', 'down').
            img_shape (tuple[int, int, int]): Image shape in (height, width, channels).

        Returns:
            end_x (int): X-coordinate of the endpoint.
            end_y (int): Y-coordinate of the endpoint.
        """
        h, w = img_shape[:2]
        if direction == "left":
            return 0, mid_y
        elif direction == "right":
            return w - 1, mid_y
        elif direction == "up":
            return mid_x, 0
        elif direction == "down":
            return mid_x, h - 1
        else:
            return mid_x, mid_y


    def draw_tracking_scope(self, im, bbox: tuple, color: tuple) -> None:
        """
        Draw tracking scope lines extending from the bounding box to image edges.

        Args:
            im (np.ndarray): Image array to draw on.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            color (tuple): Color in BGR format for drawing.
        """
        x1, y1, x2, y2 = bbox
        mid_top = ((x1 + x2) // 2, y1)
        mid_bottom = ((x1 + x2) // 2, y2)
        mid_left = (x1, (y1 + y2) // 2)
        mid_right = (x2, (y1 + y2) // 2)
        cv2.line(im, mid_top, self.extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
        cv2.line(im, mid_bottom, self.extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
        cv2.line(im, mid_left, self.extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
        cv2.line(im, mid_right, self.extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


    def click_event(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        Handle mouse click events to select an object for focused tracking.

        Args:
            event (int): OpenCV mouse event type.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (Any): Additional parameters (not used).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            detections = self.results[0].boxes.data if self.results[0].boxes is not None else []
            if detections is not None:
                min_area = float("inf")
                best_match = None
                for track in detections:
                    track = track.tolist()
                    if len(track) >= 6:
                        x1, y1, x2, y2 = map(int, track[:4])
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            area = (x2 - x1) * (y2 - y1)
                            if area < min_area:
                                class_id = int(track[-1])
                                track_id = int(track[4]) if len(track) == 7 else -1
                                min_area = area
                                best_match = (track_id, self.model.names[class_id])
                if best_match:
                    self.selected_object_id, label = best_match
                    print(f"🔵 TRACKING STARTED: {label} (ID {self.selected_object_id})")
                    
    def close_camera(self):
        self.cap.release()
        if self.save_video and self.vw is not None:
            self.vw.release()
        cv2.destroyAllWindows()
        
    def exec(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_event)

        fps_counter, fps_timer, fps_display = 0, time.time(), 0

        while self.cap.isOpened():
            success, im = self.cap.read()
            if not success:
                break

            self.results = self.model.track(im, conf=self.conf, iou=self.iou, max_det=self.max_det, tracker=self.tracker, classes=self.target_classes, **self.track_args)
            annotator = Annotator(im)
            detections = self.results[0].boxes.data if self.results[0].boxes is not None else []
            detected_objects = []
            for track in detections:
                track = track.tolist()
                if len(track) < 6:
                    continue
                x1, y1, x2, y2 = map(int, track[:4])
                class_id = int(track[6]) if len(track) >= 7 else int(track[5])
                track_id = int(track[4]) if len(track) == 7 else -1
                color = colors(track_id, True)
                txt_color = annotator.get_txt_color(color)
                label = f"{self.classes[class_id]} ID {track_id}" + (f" ({float(track[5]):.2f})" if self.show_conf else "")
                if track_id == self.selected_object_id:
                    self.draw_tracking_scope(im, (x1, y1, x2, y2), color)
                    center = self.get_center(x1, y1, x2, y2)
                    cv2.circle(im, center, 6, color, -1)

                    # Pulsing circle for attention
                    pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
                    cv2.circle(im, center, pulse_radius, color, 2)

                    annotator.box_label([x1, y1, x2, y2], label=f"ACTIVE: TRACK {track_id}", color=color)
                else:
                    # Draw dashed box for other objects
                    for i in range(x1, x2, 10):
                        cv2.line(im, (i, y1), (i + 5, y1), color, 3)
                        cv2.line(im, (i, y2), (i + 5, y2), color, 3)
                    for i in range(y1, y2, 10):
                        cv2.line(im, (x1, i), (x1, i + 5), color, 3)
                        cv2.line(im, (x2, i), (x2, i + 5), color, 3)
                    # Draw label text with background
                    (tw, th), bl = cv2.getTextSize(label, 0, 0.7, 2)
                    cv2.rectangle(im, (x1 + 5 - 5, y1 + 20 - th - 5), (x1 + 5 + tw + 5, y1 + 20 + bl), color, -1)
                    cv2.putText(im, label, (x1 + 5, y1 + 20), 0, 0.7, txt_color, 1, cv2.LINE_AA)

            if self.show_fps:
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                # Draw FPS text with background
                fps_text = f"FPS: {fps_display}"
                cv2.putText(im, fps_text, (10, 25), 0, 0.7, (255, 255, 255), 1)
                (tw, th), bl = cv2.getTextSize(fps_text, 0, 0.7, 2)
                cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
                cv2.putText(im, fps_text, (10, 25), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

            cv2.imshow(self.window_name, im)
            if self.save_video and self.vw is not None:
                self.vw.write(im)
            # Terminal logging
            LOGGER.info(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                LOGGER.info("🟢 TRACKING RESET")
                self.selected_object_id = None
        self.close_camera()

if __name__ == '__main__':
    tracker = Tracker()
    tracker.exec()
    sys.exit(0)