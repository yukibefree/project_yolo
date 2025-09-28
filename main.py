import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from tracker.tracker import Tracker
import numpy as np
from routers import root, rest_api, websocket
from app_globals import globals

# プロジェクトルートの設定
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 動画像設定
url = 'https://www.youtube.com/watch?v=XBnob1Mps4g'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    アプリケーションのライフサイクルを管理するコンテキストマネージャ。
    サーバー起動時にトラッカーを初期化し、終了時にリソースを解放します。
    """
    # サーバー起動時の処理
    globals.tracker = Tracker()
    globals.ws_tracker = Tracker()
    print("Tracker initialized.")
    yield
    # サーバー終了時の処理
    if globals.tracker:
        globals.tracker.release_capture()
        print("Tracker resources released.")
    if globals.ws_tracker:
        globals.ws_tracker.release_capture()
        print("Tracker resources released.")

# FastAPIアプリケーションのインスタンス
app = FastAPI(lifespan=lifespan)

# static を静的ファイルとしてマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2Templatesの設定
app.state.templates = Jinja2Templates(directory="templates")

# ルーティングの設定
app.include_router(root.router)
app.include_router(rest_api.router)
app.include_router(websocket.router)

# アプリケーションの実行
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )