@echo off
:: PixelHDM-RPEA-DinoV3 - Activate Virtual Environment

set "PROJECT_ROOT=%~dp0.."
set "VENV_PATH=%PROJECT_ROOT%\.venv\pixelHDM-env"

if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found
    echo         Please run: scripts\setup_env.bat
    pause
    exit /b 1
)

call "%VENV_PATH%\Scripts\activate.bat"
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%"

echo PixelHDM-RPEA-DinoV3 environment activated
echo Working directory: %PROJECT_ROOT%
