@echo off
setlocal EnableDelayedExpansion

:: Set paths
set "PROJECT_ROOT=%~dp0.."
set "VENV_PATH=%PROJECT_ROOT%\.venv\pixelHDM-env"

:: Check virtual environment
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found
    echo         Please run: scripts\setup_env.bat
    pause
    exit /b 1
)

:: Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%"

echo ========================================
echo PixelHDM-RPEA-DinoV3 - Training
echo ========================================
echo.

:: Default config
set "CONFIG=configs/train_config.yaml"

:: Parse arguments
:parse_args
if "%~1"=="" goto run
if "%~1"=="--config" (
    set "CONFIG=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--resume" (
    set "RESUME=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run
echo Config file: %CONFIG%
if defined RESUME echo Resume from: %RESUME%
echo.

:: Run training
if defined RESUME (
    python -m src.training.train --config "%CONFIG%" --resume "%RESUME%"
) else (
    python -m src.training.train --config "%CONFIG%"
)

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed
    pause
    exit /b 1
)

echo.
echo Training completed
pause
