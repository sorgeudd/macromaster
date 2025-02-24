@echo off
echo ===============================
echo Sound Macro Recorder Setup
echo ===============================

REM Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed!
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo This might take a few minutes...

REM Install core dependencies
pip install numpy
if errorlevel 1 goto error

pip install pyaudio
if errorlevel 1 goto error

pip install opencv-python
if errorlevel 1 goto error

pip install pillow
if errorlevel 1 goto error

pip install pyautogui
if errorlevel 1 goto error

echo.
echo ===================================
echo Installation Complete!
echo ===================================
echo.
echo To start the Sound Macro Recorder:
echo Run 'simple_start.bat'
echo.
echo Note: Make sure to run as Administrator
echo for proper sound and keyboard control
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
echo.
pause
exit /b 1
