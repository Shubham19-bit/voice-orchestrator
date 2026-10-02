@echo off
REM Needs DEEPGRAM_API_KEY and LLM_API_KEY in .env. Use headphones!
cd /d "%~dp0"
call .venv\Scripts\activate
python -m vox.cli live --strategy hybrid --decider heuristic-local
pause
