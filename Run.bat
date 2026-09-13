@echo off
setlocal

if not defined UV_SYSTEM_CERTS set "UV_SYSTEM_CERTS=true"

if not defined UV_DEFAULT_INDEX (
    if defined PIP_INDEX_URL (
        set "UV_DEFAULT_INDEX=%PIP_INDEX_URL%"
    ) else (
        for /f "tokens=2 delims='" %%I in ('python -m pip config list 2^>nul ^| findstr /b /c:"global.index-url="') do set "UV_DEFAULT_INDEX=%%I"
    )
)

uv run src/app.py