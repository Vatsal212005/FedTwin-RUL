@echo off
REM Runs the ESP32 bridge and restarts it automatically if it ever stops.
REM Close this window to stop the bridge.
cd /d "%~dp0.."
:loop
python hardware\rul_bridge.py --port auto
echo.
echo [bridge] stopped - restarting in 3 seconds (close this window to stop)
timeout /t 3 /nobreak >nul
goto loop
