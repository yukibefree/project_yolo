from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, HTMLResponse

from app_globals import globals

router = APIRouter()

# =======================================================
# HTTPストリーミングエンドポイント
# =======================================================
@router.get("/rest", response_class=HTMLResponse)
async def serve_rest_api_page(request: Request):
    return request.app.state.templates.TemplateResponse("rest_api.html", {"request": request})

@router.get("/raw_video_feed")
async def start_raw_video_feed():
    if globals.tracker:
        return StreamingResponse(globals.tracker.exec(raw=True), media_type="multipart/x-mixed-replace;boundary=frame")
    return {"message": "Tracker not initialized"}

@router.get("/processed_video_feed")
async def start_tracking_process():
    if globals.tracker:
        return StreamingResponse(globals.tracker.exec(), media_type="multipart/x-mixed-replace;boundary=frame")
    return {"message": "Tracker not initialized"}