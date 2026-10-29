@echo off
REM ── Local AI Interview Assistant ── launcher script ──

echo ============================================
echo   Local AI Interview Assistant
echo ============================================
echo.

REM Try to activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [WARN] Virtual environment not found at venv\
    echo        Create one with:  python -m venv venv
    echo        Then install deps: pip install -r requirements.txt
    echo.
)

REM Determine python executable
set "PYTHON_EXE=python"
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    echo [OK] Using virtual environment Python: venv\Scripts\python.exe
)

REM Check PySide6 is importable
%PYTHON_EXE% -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PySide6 is not installed in the active environment.
    echo         Run:  pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Launching application...
echo.
%PYTHON_EXE% -m app.main
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
