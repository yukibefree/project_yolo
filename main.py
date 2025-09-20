import os
import sys
from fastapi import FastAPI

from tracker.tracker import Tracker

# プロジェクトルートの設定
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# FastAPIアプリケーションのインスタンス
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "YOLO Real-time Tracking API"}

@app.get("/status")
def get_status():
    return {"status": "ok"}


# TODO: WebSocketエンドポイントとトラッキングロジックを追加