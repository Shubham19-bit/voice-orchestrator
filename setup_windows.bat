@echo off
REM One-time setup on Windows: creates a virtual env and installs dependencies.
cd /d "%~dp0"
python -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup done. Next: run_demo.bat
pause
