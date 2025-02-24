@echo off
echo Starting Sound Macro Recorder...

REM Change to the script's directory
cd /d "%~dp0"

REM Check if Python is in PATH
python --version >nul 2>nul
if errorlevel 1 (
    echo Error: Python not found
    echo Please run simple_install.bat first
    pause
    exit /b 1
)

REM Run the application
python gui_interface.py
if errorlevel 1 (
    echo Error: Failed to start application
    echo Please ensure all dependencies are installed correctly
    pause
    exit /b 1
)

pause
