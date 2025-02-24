@echo off
echo Starting Sound Macro Application...

REM Change to the script's directory
cd /d "%~dp0"

REM Check if Python is in PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found in PATH
    echo Please run install.bat first to set up Python and dependencies
    pause
    exit /b 1
)

REM Run the application
python gui_interface.py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to start application
    echo Please ensure all dependencies are installed by running install.bat
    pause
    exit /b 1
)

pause