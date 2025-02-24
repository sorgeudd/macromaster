# Fishing Bot

A Python-based fishing bot with GUI controls for online game automation.

## Requirements

- Windows operating system
- Python 3.8 or later
- Administrator privileges (for keyboard control)

## Quick Installation

1. Extract all files from the zip archive
2. Run `install.bat` as Administrator
3. Follow the on-screen instructions

## Manual Installation

If the automatic installation fails, you can install the required packages manually:

```bash
pip install numpy opencv-python pillow pyaudio pyautogui requests torch transformers
```

## Testing Instructions

### Mouse Movement Testing
1. Run the bot as Administrator:
   ```bash
   python fishing_bot.py
   ```
2. Enter any window title in the "Window Title" field
3. Click "Detect Window"
4. Click "Start Recording" in the Macro section
5. Move your mouse within the window
6. Click "Stop Recording"
7. Play back the macro to verify mouse movement

Expected behavior:
- Mouse should move smoothly following recorded path
- Coordinates should be relative to window position
- No leftward drift or getting stuck at edges

### Sound Trigger Testing
1. Click "Sound Triggers" tab
2. Enter a trigger name (e.g., "test")
3. Select action type (e.g., "Key Press")
4. Click "Start Sound Triggers"
5. Make a sound
6. Click "Stop Sound Triggers"

Expected behavior:
- "Sound Triggers Active" status when started
- "Sound Triggers Stopped" status when stopped
- Action executes when sound detected

## Usage
1. Run the bot:
   - Double click on `start_bot.bat`, OR
   - Run `python fishing_bot.py` in terminal
2. Configure the detection area and key bindings in the GUI
3. Start learning mode:
   - Click "Start Learning"
   - Perform the fishing actions you want the bot to learn
   - Click "Stop Learning" when done
4. Click "Start Bot" to begin automated fishing
5. Click "Stop" to pause the bot

## Configuration

- Detection Area: The screen region to monitor for fish bites (format: x,y,width,height)
- Cast Key: The key to press for casting the fishing line (default: 'f')
- Reel Key: The key to press for reeling in fish (default: 'r')

## Game Window Detection

1. Enter the game window title (or leave blank for auto-detection)
2. Click "Detect Window" to find and focus the game window
3. The bot will automatically track the selected window

## Map Management

- Load local map files (JSON or CSV format)
- Download maps from URLs for navigation and resource locations
- Maps are used for pathfinding and resource detection

## Learning Mode

The learning mode allows the bot to learn from your actions:

1. Click "Start Learning" button
2. Perform your normal fishing routine:
   - Cast your line
   - Wait for fish
   - Reel in when you get a bite
3. Click "Stop Learning" when you're satisfied
4. The bot will analyze and learn from your actions
5. Start the bot to use the learned patterns

## Features

- AI-powered pattern recognition
- Automatic window detection
- Map-based navigation
- Custom learning mode
- Emergency stop (F6)
- Detailed logging
- Configurable key bindings
- Sound-triggered actions
- Macro recording and playback

## Troubleshooting

1. If mouse movement issues persist:
   - Verify running as Administrator
   - Check window detection status
   - Review debug logs for coordinate translation

2. If sound triggers don't work:
   - Check audio input device settings
   - Verify PyAudio installation
   - Review sound trigger status in logs

3. For general issues:
   - Run as Administrator
   - Check all dependencies are installed
   - Review logs in `fishing_bot_[DATE].log`

## Project Structure

```
fishing_bot/
├── bot_core.py          # Core bot functionality
├── gui_components.py    # GUI implementation
├── fishing_bot.py       # Main entry point
├── gameplay_learner.py  # AI learning system
├── vision_system.py     # Computer vision components
├── config_manager.py    # Configuration handling
├── logger.py           # Logging setup
├── models/             # AI model files
└── attached_assets/    # Resource files
```

## GitHub Repository

The project is available on GitHub. To use:

1. Clone or download the repository
2. Follow the installation instructions above
3. Run the bot using `python fishing_bot.py`

## Support

For issues and feature requests, please create an issue on GitHub.