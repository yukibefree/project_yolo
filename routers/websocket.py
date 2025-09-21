from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app_globals import globals

router = APIRouter()

# =======================================================
# WebSocketエンドポイント
# =======================================================
@router.get("/ws", response_class=HTMLResponse)
async def serve_websocket_page(request: Request):
    return request.app.state.templates.TemplateResponse("websocket.html", {"request": request})

@router.websocket("/ws_raw_video_feed")
async def start_ws_raw_video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_bytes in globals.ws_tracker.ws_exec(raw=True):
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        print("クライアントが切断しました。")

@router.websocket("/ws_processed_video_feed")
async def start_ws_processed_video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_bytes in globals.ws_tracker.ws_exec():
            await websocket.send_bytes(frame_bytes)
    except WebSocketDisconnect:
        print("クライアントが切断しました。")