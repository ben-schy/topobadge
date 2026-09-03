@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up topobadge for the first time - this only happens once...
    python -m venv .venv
    if errorlevel 1 (
        echo Could not find Python. Install Python 3.11+ from https://python.org and try again.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .
)

".venv\Scripts\python.exe" -m topobadge serve
pause
