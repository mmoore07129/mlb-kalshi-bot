@echo off
cd /d "%~dp0"
echo ==========================================
echo   MLB Kalshi Bot - TRAIN MODEL
echo ==========================================
echo.
echo Fetching 2015-2025 game logs from MLB API and training XGBoost...
echo This takes about 20-30 minutes (pitcher game-log fetch dominates).
echo.
python models\train.py --start 2015 --end 2025
echo.
pause
