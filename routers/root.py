from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app_globals import globals

router = APIRouter()

# =======================================================
# ルーティング
# =======================================================
@router.get("/")
async def serve_landing_page(request: Request):
    return request.app.state.templates.TemplateResponse("index.html", {"request": request})

@router.get("/status")
def get_status():
    if globals.tracker and globals.tracker.cap.isOpened():
        return {"status": "ok", "camera_status": "open"}
    return {"status": "ok", "camera_status": "closed"}

# =======================================================
# 比較用
# =======================================================
@router.get("/compare", response_class=HTMLResponse)
async def serve_compare_page(request: Request):
    return request.app.state.templates.TemplateResponse("compare.html", {"request": request})