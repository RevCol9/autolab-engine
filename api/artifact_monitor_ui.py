"""Static browser console for the model artifact monitor."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["model-artifact-monitor-ui"])

_ASSET_ROOT = Path(__file__).resolve().parent / "static" / "artifact_monitor"
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


@router.get("/monitor/model-artifacts", include_in_schema=False)
def artifact_monitor_page() -> FileResponse:
    return FileResponse(
        _ASSET_ROOT / "index.html",
        media_type="text/html",
        headers={
            **_NO_STORE_HEADERS,
            "Content-Security-Policy": (
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; img-src 'self' data:"
            ),
        },
    )


@router.get("/monitor/model-artifacts/assets/{asset_name}", include_in_schema=False)
def artifact_monitor_asset(asset_name: str) -> FileResponse:
    assets = {
        "app.js": ("application/javascript", _ASSET_ROOT / "app.js"),
        "styles.css": ("text/css", _ASSET_ROOT / "styles.css"),
    }
    if asset_name not in assets:
        raise HTTPException(status_code=404, detail="监控页面资源不存在")
    media_type, path = assets[asset_name]
    return FileResponse(path, media_type=media_type, headers=_NO_STORE_HEADERS)


__all__ = ["artifact_monitor_page", "router"]
