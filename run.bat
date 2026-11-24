@echo off
rem Mafia agent terminal launcher: run.bat [watch|play|replay|ask] [--seed N] [--quiet]
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" play.py %*
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    "%~dp0..\venv\Scripts\python.exe" play.py %*
) else (
    python play.py %*
)
