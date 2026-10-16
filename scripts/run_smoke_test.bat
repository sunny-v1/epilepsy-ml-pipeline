@echo off
REM Create .venv with Python 3.15 (if missing), install dependencies, and run the smoke test.
SETLOCAL
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment with Python 3.15...
    py -3.15 -m venv .venv || exit /b 1
)

REM PyYAML has no Python 3.15 wheel yet; build its pure-Python version without Cython.
set PYYAML_FORCE_CYTHON=0
set PYYAML_FORCE_LIBYAML=0

.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python scripts\smoke_test.py

ENDLOCAL
