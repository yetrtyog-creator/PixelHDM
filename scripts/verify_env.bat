@echo off
setlocal EnableDelayedExpansion

:: Set paths
set "PROJECT_ROOT=%~dp0.."
set "VENV_PATH=%PROJECT_ROOT%\.venv\pixelHDM-env"

echo ========================================
echo PixelHDM-RPEA-DinoV3 - Environment Check
echo ========================================
echo.

:: Check virtual environment
echo [1/6] Checking virtual environment...
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo       [FAIL] Virtual environment not found
    echo       Please run: scripts\setup_env.bat
    pause
    exit /b 1
)
echo       Virtual environment exists OK

:: Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%"

:: Check Python version
echo.
echo [2/6] Checking Python version...
python -c "import sys; v=f'{sys.version_info.major}.{sys.version_info.minor}'; print(f'       Python {v}', 'OK' if v=='3.12' else 'WARN (need 3.12)')"

:: Check PyTorch
echo.
echo [3/6] Checking PyTorch...
python -c "import torch; v=torch.__version__; print(f'       PyTorch {v}', 'OK' if v.startswith('2.') else 'WARN (need 2.x)')"

:: Check CUDA
echo.
echo [4/6] Checking CUDA...
python -c "import torch; c=torch.version.cuda or 'N/A'; avail=torch.cuda.is_available(); print(f'       CUDA {c}', 'OK' if avail else 'WARN')"
python -c "import torch; print(f'       GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

:: Check Flash Attention
echo.
echo [5/6] Checking Flash Attention...
python -c "from flash_attn import flash_attn_func; print('       Flash Attention OK')" 2>nul
if errorlevel 1 (
    echo       Flash Attention WARN (not installed)
)

:: Check project imports
echo.
echo [6/6] Checking project imports...
python -c "from src.config import PixelHDMConfig; print('       Config OK')" 2>nul
if errorlevel 1 (
    echo       [FAIL] Config import failed
)
python -c "from src.models.pixelhdm import PixelHDM; print('       Model OK')" 2>nul
if errorlevel 1 (
    echo       [FAIL] Model import failed
)
python -c "from src.inference.pipeline import PixelHDMPipeline; print('       Pipeline OK')" 2>nul
if errorlevel 1 (
    echo       [FAIL] Pipeline import failed
)

:: Detailed info
echo.
echo ========================================
echo Detailed Information
echo ========================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')" 2>nul
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Current GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name()}')" 2>nul

echo.
echo ========================================
echo Installed Packages
echo ========================================
pip list | findstr /i "torch flash transformers accelerate bitsandbytes triton"

echo.
pause
