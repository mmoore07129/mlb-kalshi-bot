@echo off
cd /d "%~dp0"
echo ==========================================
echo   MLB Kalshi Bot - DRY RUN (no real bets)
echo ==========================================
echo.
python main.py --dry-run
echo.
pause
