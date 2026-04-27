@echo off
cd /d "%~dp0"
echo ==========================================
echo   MLB Kalshi Bot - LIVE + WATCH MODE
echo   Places orders, then re-prices until games start
echo ==========================================
echo.
python main.py --watch
echo.
pause
