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

:: Default values (set once)
set "CHECKPOINT="
set "OUTPUT_DIR=outputs"
set "NUM_STEPS=50"
set "CFG_SCALE=7.5"
set "WIDTH=512"
set "HEIGHT=512"
set "SEED="
set "USE_EMA=1"
set "SAMPLER=heun"

:main_menu
cls
echo ========================================
echo PixelHDM-RPEA-DinoV3 - Inference
echo ========================================
echo.
echo [NOTE] First run will load text encoder
echo.

echo Current settings:
echo   [1] Steps:     %NUM_STEPS%
echo   [2] CFG:       %CFG_SCALE%
echo   [3] Size:      %WIDTH%x%HEIGHT%
if "%SEED%"=="" (echo   [4] Seed:      [random]) else (echo   [4] Seed:      %SEED%)
if "%USE_EMA%"=="1" (echo   [5] Weights:   EMA) else (echo   [5] Weights:   Model)
echo   [6] Sampler:   %SAMPLER%
if "%CHECKPOINT%"=="" (echo   [7] Checkpoint: [auto-detect]) else (echo   [7] Checkpoint: %CHECKPOINT%)
echo.
echo   [G] Generate image
echo   [Q] Quit
echo.

set /p "CHOICE=Select option or enter prompt: "

if /i "%CHOICE%"=="Q" goto end
if /i "%CHOICE%"=="G" goto get_prompt
if "%CHOICE%"=="1" goto set_steps
if "%CHOICE%"=="2" goto set_cfg
if "%CHOICE%"=="3" goto set_size
if "%CHOICE%"=="4" goto set_seed
if "%CHOICE%"=="5" goto toggle_ema
if "%CHOICE%"=="6" goto set_sampler
if "%CHOICE%"=="7" goto set_checkpoint

:: If not a menu option, treat as prompt
set "PROMPT=%CHOICE%"
goto generate

:set_steps
set /p "NUM_STEPS=Steps (10-100): "
goto main_menu

:set_cfg
set /p "CFG_SCALE=CFG scale (1.0-15.0): "
goto main_menu

:set_size
echo Presets: 512x512, 768x768, 1024x768, 768x1024
echo (Must be multiples of 16)
echo.
set /p "WIDTH=Width: "
set /p "HEIGHT=Height: "
goto main_menu

:set_seed
set /p "SEED=Seed (empty=random): "
goto main_menu

:toggle_ema
if "%USE_EMA%"=="1" (set "USE_EMA=0") else (set "USE_EMA=1")
goto main_menu

:set_sampler
echo Sampler options: euler, heun, dpm_pp
set /p "SAMPLER=Sampler: "
goto main_menu

:set_checkpoint
echo Available checkpoints:
dir /b checkpoints\*.pt 2>nul
echo.
echo (Leave empty to auto-detect latest)
set /p "CHECKPOINT=Checkpoint path: "
goto main_menu

:get_prompt
set /p "PROMPT=Prompt: "
if "%PROMPT%"=="" (
    echo [ERROR] Prompt cannot be empty
    pause
    goto main_menu
)

:generate
echo.
echo ----------------------------------------
echo Generating...
echo   Prompt: %PROMPT%
echo   Steps: %NUM_STEPS%, CFG: %CFG_SCALE%, Size: %WIDTH%x%HEIGHT%
if not "%SEED%"=="" echo   Seed: %SEED%
if "%USE_EMA%"=="1" (echo   Weights: EMA) else (echo   Weights: Model)
echo   Sampler: %SAMPLER%
echo ----------------------------------------
echo.

:: Build optional arguments
set "SEED_ARG="
if not "%SEED%"=="" set "SEED_ARG=--seed %SEED%"

set "EMA_ARG="
if "%USE_EMA%"=="0" set "EMA_ARG=--no-ema"

set "CKPT_ARG="
if not "%CHECKPOINT%"=="" set "CKPT_ARG=--checkpoint %CHECKPOINT%"

:: Run inference (checkpoint auto-detected if not specified)
python -m src.inference.run %CKPT_ARG% --prompt "%PROMPT%" --output "%OUTPUT_DIR%" --steps %NUM_STEPS% --cfg %CFG_SCALE% --width %WIDTH% --height %HEIGHT% --sampler %SAMPLER% %SEED_ARG% %EMA_ARG%

if errorlevel 1 (
    echo.
    echo [ERROR] Inference failed
    pause
    goto main_menu
)

echo.
echo [OK] Saved to: %OUTPUT_DIR%
echo.
pause
goto main_menu

:end
echo Goodbye.
