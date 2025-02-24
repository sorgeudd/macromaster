# Sound Macro Recorder

A Python-based sound macro recorder that lets you trigger mouse and keyboard macros with sound.

## Features

- Record sound triggers
- Record mouse and keyboard macros
- Map sounds to macros
- Easy-to-use GUI interface
- Real-time sound monitoring
- Visual macro playback

## Requirements

- Windows operating system
- Python 3.8 or later
- Microphone for sound recording
- Administrator privileges (for keyboard control)

## Quick Installation

1. Extract all files from the zip archive
2. Run `install.bat` as Administrator
3. Follow the on-screen instructions

## Manual Installation

If the automatic installation fails, you can install the required packages manually:

```bash
pip install numpy opencv-python pillow pyaudio pyautogui
```

## Usage

1. Run the application:
   - Double click on `start_app.bat`, OR
   - Run `python gui_interface.py` in terminal

2. Record a sound trigger:
   - Enter a name for your sound trigger
   - Click "Record Sound"
   - Make the sound you want to use as a trigger

3. Record a macro:
   - Enter a name for your macro
   - Click "Record Macro"
   - Perform the mouse/keyboard actions you want to record
   - The recording will stop automatically after 5 seconds

4. Map sound to macro:
   - Select your recorded sound and macro from the dropdown menus
   - Click "Map" to link them together

5. Start monitoring:
   - Click "Start Monitoring"
   - Make your recorded sound to trigger the macro
   - Click "Stop Monitoring" when done

## Troubleshooting

1. If sound recording doesn't work:
   - Check your microphone settings
   - Verify PyAudio installation
   - Run the application as Administrator

2. If macro playback isn't working:
   - Run the application as Administrator
   - Check if the target window is active
   - Verify the macro was recorded correctly

3. For general issues:
   - Check all dependencies are installed
   - Run as Administrator
   - Review logs in the status bar

## Support

For issues and feature requests, please reach out through the provided support channels.