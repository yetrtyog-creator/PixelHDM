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
echo PixelHDM-RPEA-DinoV3 - Run Tests
echo ========================================
echo.

:: Parse arguments
set "TEST_PATH=tests/"
set "VERBOSE="
set "COVERAGE="

:parse_args
if "%~1"=="" goto run
if "%~1"=="-v" (
    set "VERBOSE=-v"
    shift
    goto parse_args
)
if "%~1"=="--verbose" (
    set "VERBOSE=-v"
    shift
    goto parse_args
)
if "%~1"=="--cov" (
    set "COVERAGE=--cov=src --cov-report=term-missing"
    shift
    goto parse_args
)
if "%~1"=="--path" (
    set "TEST_PATH=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run
echo Test path: %TEST_PATH%
echo.

:: Run pytest
pytest %TEST_PATH% %VERBOSE% %COVERAGE%

if errorlevel 1 (
    echo.
    echo [WARN] Some tests failed
) else (
    echo.
    echo [OK] All tests passed
)

pause
