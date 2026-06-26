@echo off
echo.
echo  Starting AES-GCM UI...
echo  Open your browser at: http://localhost:8080
echo.

REM Open the browser automatically after a short delay
start "" timeout /t 1 /nobreak >nul
start "" "http://localhost:8080"

python "%~dp0server.py"
