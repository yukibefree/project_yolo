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

# インスタンス用変数
tracker = None
ws_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリケーションのライフサイクルを管理するコンテキストマネージャ。
    サーバー起動時にトラッカーを初期化し、終了時にリソースを解放します。
    """
    # サーバー起動時の処理
    global tracker
    global ws_tracker
    if tracker is None:
        tracker = Tracker()
    if ws_tracker is None:
        ws_tracker = Tracker()
    print("Tracker initialized.")
    yield
    # サーバー終了時の処理
    if tracker:
        tracker.close_camera()
        print("Tracker resources released.")
    if ws_tracker:
        ws_tracker.close_camera()
        print("Tracker resources released.")

# FastAPIアプリケーションのインスタンス
app = FastAPI(lifespan=lifespan)

# static を静的ファイルとしてマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2Templatesの設定
templates = Jinja2Templates(directory="templates")

# =======================================================
# ルーティング
# =======================================================
@app.get("/", response_class=HTMLResponse)
async def serve_page(request: Request):
    # 映像を表示するHTMLページを返す
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
def get_status():
    if tracker and tracker.cap.isOpened():
        return {"status": "ok", "camera_status": "open"}
    return {"status": "ok", "camera_status": "closed"}

# =======================================================
# 比較用
# =======================================================
@app.get("/compare", response_class=HTMLResponse)
async def serve_compare_page(request: Request):
    # 映像を表示するHTMLページを返す
    return templates.TemplateResponse("compare.html", {"request": request})

# =======================================================
# HTTPストリーミングエンドポイント
# =======================================================
@app.get("/rest", response_class=HTMLResponse)
async def serve_rest_api_page(request: Request):
    # 映像を表示するHTMLページを返す
    return templates.TemplateResponse("rest_api.html", {"request": request})

# 処理前の元映像
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
# WebSocketエンドポイント
# =======================================================
@app.get("/ws", response_class=HTMLResponse)
async def serve_websocket_page(request: Request):
    return templates.TemplateResponse("websocket.html", {"request": request})

# 処理前の元映像
@app.websocket("/ws_raw_video_feed")
async def start_ws_raw_video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_bytes in ws_tracker.ws_exec(raw=True):
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        print("クライアントが切断しました。")

@app.websocket("/ws_processed_video_feed")
async def start_ws_processed_video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_bytes in ws_tracker.ws_exec():
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