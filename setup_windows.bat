@echo off
REM ============================================
REM Mirage-in-the-Eyes Windows 11 One-Click Setup Script
REM For: RTX 4060 Laptop (8GB VRAM)
REM Plan A: Minimal MiniGPT-4 Reproduction
REM ============================================

REM Check if mllm environment exists before removing
conda env list | findstr /C:"mllm" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Found existing mllm environment, removing...
    conda remove --prefix "C:\Users\25646\.conda\envs\mllm" --all -y
) else (
    echo [INFO] No existing mllm environment found, skipping removal
)
echo ============================================
echo Mirage-in-the-Eyes Windows 11 Setup
echo Hardware: RTX 4060 Laptop (8GB VRAM)
echo Plan: A - Minimal Reproduction
echo ============================================
echo.

REM Check if running in correct directory
if not exist "requirements_windows.txt" (
    echo [ERROR] Please run this script in project root: d:\AI PROJEAT\mirage
    pause
    exit /b 1
)

echo [Step 1/7] Creating conda environment mllm...
call conda create -n mllm python=3.9.20 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)
echo [DONE] Conda environment created successfully
echo.

echo [Step 2/7] Activating conda environment...
call conda activate mllm
if errorlevel 1 (
    echo [WARNING] Auto-activation failed, please run manually: conda activate mllm
)
echo.

echo [Step 3/7] Installing PyTorch (CUDA 11.8)...
echo Note: This may take several minutes...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    pause
    exit /b 1
)
echo [DONE] PyTorch installed successfully
echo.

echo [Step 4/7] Installing project dependencies...
echo Note: This may take 10-20 minutes...
pip install -r requirements_windows.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies failed, but core functionality may still work
)
echo [DONE] Dependencies installation completed
echo.

echo [Step 5/7] Installing CLIP...
pip install git+https://github.com/openai/CLIP.git
if errorlevel 1 (
    echo [ERROR] CLIP installation failed, please check Git and network connection
    pause
    exit /b 1
)
echo [DONE] CLIP installed successfully
echo.

echo [Step 6/7] Installing baukit...
pip install git+https://github.com/davidbau/baukit.git
if errorlevel 1 (
    echo [WARNING] baukit installation failed, some features may be unavailable
)
echo.

echo [Step 7/7] Extracting and installing local transformers...
if exist "transformers-4.29.2" (
    echo [INFO] transformers-4.29.2 directory exists, skipping extraction
) else (
    powershell -command "Expand-Archive -Force transformers-4.29.2.zip ."
    if errorlevel 1 (
        echo [ERROR] Failed to extract transformers
        pause
        exit /b 1
    )
)

cd transformers-4.29.2
python -m pip install -e .
if errorlevel 1 (
    echo [ERROR] transformers installation failed
    cd ..
    pause
    exit /b 1
)
cd ..
echo [DONE] transformers installed successfully
echo.

echo ============================================
echo Environment setup completed!
echo ============================================
echo.
echo Next steps:
echo 1. Verify installation: python verify_install.py
echo 2. Download model weights (see implementation_plan.md)
echo 3. Configure model paths (edit eval_configs/*.yaml)
echo.
echo Note: Since your GPU only has 8GB VRAM, we recommend:
echo   - Use only Vicuna-7B (do not use 13B)
echo   - Close other programs using GPU memory
echo   - Use batch_size=1
echo.
pause
