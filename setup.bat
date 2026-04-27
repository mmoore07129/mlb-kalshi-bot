@echo off
cd /d "%~dp0"
echo Installing MLB Kalshi Bot dependencies (native Python)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Run run_dry.bat first to test, then run.bat to go live.
pause
