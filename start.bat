@echo off
REM =====================================================================
REM  SENTIO — Full-Stack Development Launcher
REM  Boots FastAPI backend (port 8000) + Vite React frontend (port 5173)
REM =====================================================================

title SENTIO Launcher
echo.
echo  ======================================================
echo    SENTIO: Contactless Affective Computing Framework
echo    Full-Stack Dashboard Launcher
echo  ======================================================
echo.

REM ── Resolve paths ──────────────────────────────────────────
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv\Scripts"
set "FRONTEND=%ROOT%frontend"

REM ── Verify Python venv ─────────────────────────────────────
if not exist "%VENV%\python.exe" (
    echo  [ERROR] Python virtual environment not found at %VENV%
    echo          Run: python -m venv .venv
    pause
    exit /b 1
)

REM ── Verify node_modules ────────────────────────────────────
if not exist "%FRONTEND%\node_modules" (
    echo  [INFO] Installing frontend dependencies...
    cd /d "%FRONTEND%"
    call npm install
    cd /d "%ROOT%"
    echo.
)

echo  [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "SENTIO Backend" cmd /k "cd /d "%ROOT%backend" && "%VENV%\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Give the backend a moment to initialize the camera.
timeout /t 3 /nobreak > nul

echo  [2/2] Starting Vite frontend on http://localhost:5173 ...
start "SENTIO Frontend" cmd /k "cd /d "%FRONTEND%" && npm run dev"

echo.
echo  ======================================================
echo    Both servers are starting in separate windows.
echo.
echo    Frontend : http://localhost:5173
echo    Backend  : http://localhost:8000
echo    API Docs : http://localhost:8000/docs
echo  ======================================================
echo.
echo  Press any key to close this launcher (servers stay running).
pause > nul
