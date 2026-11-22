@echo off
REM FedTwin-RUL hardware demo: double-click this file.
REM 1) plug in the ESP32 first  2) close the Arduino Serial Monitor
cd /d "%~dp0"

echo Starting ESP32 bridge...
start "FedTwin-RUL bridge (keep open)" cmd /k hardware\run_bridge.bat

echo Starting dashboard...
start "FedTwin-RUL dashboard (keep open)" cmd /k "cd app\frontend && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort"

echo Waiting for the dashboard to start...
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:5173/

echo.
echo Demo running. Keep the two black windows open.
echo To stop: close both windows (the board goes OFFLINE = yellow blink).
timeout /t 8 >nul
