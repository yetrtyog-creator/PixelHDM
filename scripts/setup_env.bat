@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo PixelHDM-RPEA-DinoV3 - Environment Setup
echo ========================================
echo.

:: Set paths
set "PROJECT_ROOT=%~dp0.."
set "VENV_PATH=%PROJECT_ROOT%\.venv\pixelHDM-env"

:: Check Python 3.12 via py launcher
echo [1/7] Checking Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12 not found.
    echo.
    echo         Please install Python 3.12:
    echo         https://www.python.org/downloads/release/python-3127/
    echo.
    echo         NOTE: You can install alongside other Python versions
    echo               Just uncheck "Add to PATH" during installation
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('py -3.12 --version') do echo       %%i OK

:: Create virtual environment
echo.
echo [2/7] Creating virtual environment with Python 3.12...
if exist "%VENV_PATH%" (
    echo       Virtual environment already exists, skipping...
) else (
    py -3.12 -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo       Virtual environment created: %VENV_PATH%
)

:: Activate virtual environment
echo.
echo [3/7] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo       Virtual environment activated OK

:: Verify Python version in venv
for /f "tokens=*" %%i in ('python --version') do echo       Using: %%i

:: Upgrade pip
echo.
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       pip upgraded OK

:: Install PyTorch
echo.
echo [5/7] Installing PyTorch 2.8.0 + CUDA 12.8...
echo       This may take a few minutes, please wait...
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    echo         Please check network connection
    pause
    exit /b 1
)
echo       PyTorch 2.8.0+cu128 OK

:: Install Flash Attention
echo.
echo [6/7] Installing Flash Attention 2.8.3 (Windows prebuilt)...
echo       This may take a few minutes, please wait...
pip install https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu128torch2.8.0cxx11abiTRUE-cp312-cp312-win_amd64.whl --quiet
if errorlevel 1 (
    echo [WARNING] Flash Attention installation failed
    echo           You can install it manually later
    echo.
    set /p "CONTINUE=Continue anyway? (Y/N): "
    if /i "!CONTINUE!" neq "Y" (
        pause
        exit /b 1
    )
) else (
    echo       Flash Attention OK
)

:: Install other dependencies
echo.
echo [7/7] Installing other dependencies...
pip install -r "%PROJECT_ROOT%\requirements.txt" --quiet
if errorlevel 1 (
    echo [ERROR] Dependencies installation failed
    pause
    exit /b 1
)
echo       Dependencies OK

:: Verify environment
echo.
echo ========================================
echo Verifying environment...
echo ========================================
cd /d "%PROJECT_ROOT%"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA {torch.version.cuda}')" 2>nul
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul

:: Verify Flash Attention
python -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention verified')" 2>nul
if errorlevel 1 (
    echo [WARNING] Flash Attention not available
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Virtual environment: %VENV_PATH%
echo Python version: 3.12
echo.
echo ========================================
echo IMPORTANT: DINOv3 Weights (Manual Download Required)
echo ========================================
echo.
echo DINOv3 requires Meta AI license agreement.
echo Please download manually from official source:
echo   https://github.com/facebookresearch/dinov2
echo.
echo Place the weights file at:
echo   Dinov3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
echo.
if not exist "%PROJECT_ROOT%\Dinov3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" (
    echo [WARNING] DINOv3 weights NOT FOUND - REPA loss will be disabled
) else (
    echo [OK] DINOv3 weights found
)
echo.
echo ========================================
echo Usage
echo ========================================
echo.
echo   Activate:   scripts\activate.bat
echo   Train:      scripts\train.bat
echo   Inference:  scripts\inference.bat
echo   Verify:     scripts\verify_env.bat
echo.
pause
