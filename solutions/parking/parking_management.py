import cv2
import os
import sys
from ultralytics import solutions

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
# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parkingmanager = solutions.ParkingManagement(
    model=MODEL_FILE,  # path to model file
    json_file=JSON_FILE,  # path to parking annotations file
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    results = parkingmanager(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows