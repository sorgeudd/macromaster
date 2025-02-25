@echo off
title Fishing Bot Installer
color 0A

echo ================================
echo   Fishing Bot - Easy Install
echo ================================
echo.

REM Create installation directory
set INSTALL_DIR=%USERPROFILE%\FishingBot
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Installing Python...
    curl -L "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe" -o "%TEMP%\python-installer.exe"
    "%TEMP%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1
    del "%TEMP%\python-installer.exe"
)

REM Copy all files to installation directory
echo Copying files...
xcopy /Y /E /I "*.py" "%INSTALL_DIR%"
xcopy /Y /E /I "*.json" "%INSTALL_DIR%" 2>nul
if not exist "%INSTALL_DIR%\debug_screenshots" mkdir "%INSTALL_DIR%\debug_screenshots"

REM Install Python packages
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pyautogui keyboard mss python-dotenv

REM Create desktop shortcut
echo Creating desktop shortcut...
set SHORTCUT="%USERPROFILE%\Desktop\Fishing Bot.lnk"
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = %SHORTCUT% >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "pythonw.exe" >> CreateShortcut.vbs
echo oLink.Arguments = "%INSTALL_DIR%\main.py" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "Fishing Bot" >> CreateShortcut.vbs
echo oLink.IconLocation = "pythonw.exe,0" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

REM Create start script
echo @echo off > "%INSTALL_DIR%\start.bat"
echo title Fishing Bot >> "%INSTALL_DIR%\start.bat"
echo python main.py >> "%INSTALL_DIR%\start.bat"
echo pause >> "%INSTALL_DIR%\start.bat"

REM Check Visual C++ Redistributable (from original script)
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0" >nul 2>&1
if errorlevel 1 (
    echo Warning: Microsoft Visual C++ 2015-2022 Redistributable might be missing
    echo Please download and install from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
)


echo.
echo ================================
echo Installation Complete!
echo.
echo The Fishing Bot has been installed to:
echo %INSTALL_DIR%
echo.
echo A shortcut has been created on your desktop.
echo.
echo Press any key to exit...
pause > nul