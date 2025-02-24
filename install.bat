@echo off
echo ===============================
echo    Setting up Fishing Bot...
echo ===============================

REM Check system architecture
reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x86" > NUL && set ARCH=32BIT || set ARCH=64BIT
if %ARCH%==32BIT (
    echo Warning: 32-bit Windows detected. 64-bit is recommended for optimal performance.
    echo Some AI features might be limited on 32-bit systems.
    echo.
)

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed!
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
python -c "import sys; ver=sys.version_info; exit(0 if ver.major==3 and ver.minor>=8 else 1)" > nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.8 or later is required
    echo Current Python installation is too old
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo This might take a few minutes...

echo Installing numpy...
pip install numpy
if errorlevel 1 goto error

echo Installing OpenCV...
pip install opencv-python
if errorlevel 1 goto error

echo Installing Pillow...
pip install pillow
if errorlevel 1 goto error

echo Installing PyAutoGUI...
pip install pyautogui
if errorlevel 1 goto error

echo Installing Torch CPU...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error

echo Installing Transformers...
pip install transformers
if errorlevel 1 goto error

echo Installing Trafilatura...
pip install trafilatura
if errorlevel 1 goto error

echo Installing Twilio...
pip install twilio
if errorlevel 1 goto error

REM Check Visual C++ Redistributable
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0" >nul 2>&1
if errorlevel 1 (
    echo Warning: Microsoft Visual C++ 2015-2022 Redistributable might be missing
    echo Please download and install from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
)

REM Verify installations
echo.
echo Verifying installations...
python -c "import numpy; import cv2; import PIL; import pyautogui; import torch; import transformers; import trafilatura; import twilio" > nul 2>&1
if errorlevel 1 (
    echo Error: Some dependencies failed to install correctly
    echo Please check the error messages above and try again
    goto error
)

echo.
echo ===================================
echo    Installation Complete!
echo ===================================
echo.
echo To start the Fishing Bot:
echo 1. Double click on 'start_bot.bat'
echo    OR
echo 2. Run 'python fishing_bot.py' in terminal
echo.
echo Note: Make sure to run as Administrator
echo for proper keyboard control functionality
echo.
pause

exit /b 0

:error
echo.
echo Error during installation!
echo Please check:
echo 1. Your internet connection
echo 2. If you have administrator privileges
echo 3. If any antivirus is blocking the installation
echo 4. If Microsoft Visual C++ Build Tools are installed
echo.
pause
exit /b 1