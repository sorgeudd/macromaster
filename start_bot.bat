@echo off
echo ===================================
echo    Starting Fishing Bot...
echo ===================================

REM Change to the script's directory
cd /d "%~dp0"

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
) else (
    echo Please run this script as Administrator!
    echo Right-click the file and select "Run as Administrator"
    pause
    exit /b 1
)

REM Start the bot
python fishing_bot.py
if errorlevel 1 (
    echo Error running fishing_bot.py
    pause
    exit /b 1
)

pause