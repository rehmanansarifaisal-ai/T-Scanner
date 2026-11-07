@echo off
setlocal
cd /d "%~dp0"

REM T-Scanner targets Python 3.10.8. Prefer the Windows Python Launcher so a
REM stale generic "python"/venv PATH entry cannot select the wrong interpreter.
set "PYTHON_CMD=py -3.10"
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) and sys.version_info >= (3, 10, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    set "PYTHON_CMD=python"
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) and sys.version_info >= (3, 10, 8) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3.10.8 was not found.
        echo Install Python 3.10.8 and enable the Windows Python Launcher.
        echo Then verify with: py -3.10 --version
        echo Installed interpreters:
        py -0p
        pause
        exit /b 1
    )
)

echo Using Python 3.10:
%PYTHON_CMD% --version

echo.
echo Installing/checking dependencies in this Python 3.10 installation...
%PYTHON_CMD% -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed. Review the pip output above.
    pause
    exit /b 1
)

echo.
echo Starting T-Scanner...
%PYTHON_CMD% main.py
pause
