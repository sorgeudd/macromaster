# Game Automation Bot - Testing Interface

A Python-based game automation bot with advanced computer vision capabilities for precise resource detection and automation in Windows game environments.

## Quick Installation Guide for Windows 10

1. Download and extract testing_package.zip
2. Run install.bat as Administrator
3. Once installation completes, run start_app.bat
4. Open your web browser and go to http://localhost:5000

## Features

### Bot Control
- Start/Stop bot operation
- Emergency Stop (F6 hotkey)
- Learning mode for AI player
- Import training videos

### Macro Management
- Record new macros
- Play existing macros
- Name and save macros for reuse

### Sound Triggers
- Record sound triggers
- Bind to key presses
- Bind to mouse clicks
- Bind to existing macros

### Testing Features
1. Position Testing
   - Enter X and Y coordinates
   - Move to position testing
   - Real-time preview

2. Terrain Testing
   - Normal, water, mountain, forest terrain types
   - Tests movement speed and behavior

3. Resource Detection
   - Load fish or ore maps
   - Find nearby resources
   - Blue dots show detected resources

4. Real-World Testing
   - Arrow detection testing
   - Resource spot verification
   - Terrain calibration
   - Pathfinding system tests

## System Requirements
- Windows 10
- Python 3.8 or later
- Administrator privileges for installation
- Microsoft Visual C++ 2015-2022 Redistributable

## Troubleshooting

1. If installation fails:
   - Make sure you're running install.bat as Administrator
   - Check your internet connection
   - Verify Python is installed and in PATH
   - Install Visual C++ Redistributable if prompted

2. If application won't start:
   - Check all dependencies were installed correctly
   - Verify port 5000 is not in use
   - Run start_app.bat as Administrator

3. If game window isn't detected:
   - Make sure the game window is open
   - Verify the window name matches exactly
   - Try running the application as Administrator

## Logs
Application logs are stored in:
- testing_ui.log: Main application logs
- map_manager.log: Resource detection logs
- mock_environment.log: Environment simulation logs

For issues and feature requests, please file an issue in the GitHub repository.