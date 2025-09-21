import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from tracker.tracker import Tracker
import numpy as np


# プロジェクトルートの設定
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)


tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリケーションのライフサイクルを管理するコンテキストマネージャ。
    サーバー起動時にトラッカーを初期化し、終了時にリソースを解放します。
    """
    # サーバー起動時の処理
    global tracker
    if tracker is None:
        tracker = Tracker()
    print("Tracker initialized.")
    yield
    # サーバー終了時の処理
    if tracker:
        tracker.close_camera()
        print("Tracker resources released.")

# FastAPIアプリケーションのインスタンス
app = FastAPI(lifespan=lifespan)

# static を静的ファイルとしてマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2Templatesの設定
templates = Jinja2Templates(directory="templates")

@app.get("/status")
def get_status():
    if tracker and tracker.cap.isOpened():
        return {"status": "ok", "camera_status": "open"}
    return {"status": "ok", "camera_status": "closed"}

# =======================================================
# 既存のHTTPストリーミングエンドポイント (変更なし)
# =======================================================
@app.get("/", response_class=HTMLResponse)
async def serve_page(request: Request):
    # 映像を表示するHTMLページを返す
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/raw_video_feed")
async def start_raw_video_feed():
    if tracker:
        return StreamingResponse(tracker.exec(raw=True), media_type="multipart/x-mixed-replace;boundary=frame")
    return {"message": "Tracker not initialized"}

@app.get("/processed_video_feed")
async def start_tracking_process():
    """トラッキングを別スレッドまたはプロセスで開始"""
    if tracker:
        return StreamingResponse(tracker.exec(), media_type="multipart/x-mixed-replace;boundary=frame")
    return {"message": "Tracker not initialized"}

# =======================================================
# 新しいWebSocketエンドポイント
# =======================================================
@app.get("/ws", response_class=HTMLResponse)
async def serve_websocket_page(request: Request):
    return templates.TemplateResponse("websocket.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_bytes in tracker.ws_exec():
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        print("クライアントが切断しました。")

# アプリケーションの実行
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )