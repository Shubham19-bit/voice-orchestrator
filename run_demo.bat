@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m vox.cli demo --scenario refill
python -m vox.cli bench
python -m pytest -q
pause
