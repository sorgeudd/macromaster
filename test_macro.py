"""Test script for macro recording and playback functionality"""
import logging
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import platform
import traceback
import math
from direct_input import DirectInput

class MacroTester:
    def __init__(self, test_mode: bool = False):
        # Setup logging
        self.logger = logging.getLogger('MacroTester')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Add file handler
        fh = logging.FileHandler('macro_test.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.test_mode = test_mode or platform.system() != 'Windows'
        self.recording = False
        self.recorded_actions: List[Dict] = []
        self.window_handle = None
        self.window_rect = None

        # Initialize DirectInput
        try:
            self.direct_input = DirectInput(test_mode=self.test_mode)
            self.logger.info("DirectInput initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize DirectInput: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def find_window(self, window_title: str) -> bool:
        """Find window by title"""
        if self.test_mode:
            # Mock window for testing
            self.window_handle = 1
            self.window_rect = (0, 0, 800, 600)
            self.logger.info(f"Test mode: Using mock window {window_title}")
            return True

        try:
            import win32gui
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if window_title.lower() in title.lower():
                        self.window_handle = hwnd
                        self.window_rect = win32gui.GetWindowRect(hwnd)
                        return False
                return True

            win32gui.EnumWindows(callback, None)

            if self.window_handle:
                self.logger.info(f"Found window '{window_title}' at {self.window_rect}")
                return True

            self.logger.warning(f"Window '{window_title}' not found")
            return False
        except ImportError:
            self.logger.error("win32gui not available, running in test mode")
            self.test_mode = True
            return self.find_window(window_title)
        except Exception as e:
            self.logger.error(f"Error finding window: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def start_recording(self) -> bool:
        """Start recording mouse movements"""
        if not self.window_handle and not self.test_mode:
            self.logger.error("No window selected for recording")
            return False

        self.recording = True
        self.recorded_actions = []
        self.logger.info("Started recording mouse movements")
        return True

    def record_action(self, action_type: str, x: int, y: int) -> None:
        """Record a mouse action"""
        if not self.recording:
            return

        try:
            # Convert to window-relative coordinates
            if self.window_rect:
                rel_x = x - self.window_rect[0]
                rel_y = y - self.window_rect[1]
            else:
                rel_x, rel_y = x, y

            action = {
                'type': action_type,
                'x': rel_x,
                'y': rel_y,
                'timestamp': time.time()
            }

            self.recorded_actions.append(action)
            self.logger.debug(f"Recorded action: {action}")
        except Exception as e:
            self.logger.error(f"Error recording action: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")

    def stop_recording(self) -> bool:
        """Stop recording and save actions"""
        if not self.recording:
            return False

        try:
            self.recording = False

            # Save recorded actions
            with open('recorded_macro.json', 'w') as f:
                json.dump(self.recorded_actions, f, indent=2)
            self.logger.info(f"Saved {len(self.recorded_actions)} actions to recorded_macro.json")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save recorded actions: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def play_macro(self, macro_file: str = 'recorded_macro.json') -> bool:
        """Play back recorded macro"""
        if not self.window_handle and not self.test_mode:
            self.logger.error("No window selected for playback")
            return False

        try:
            # Load macro file
            with open(macro_file, 'r') as f:
                actions = json.load(f)

            self.logger.info(f"Loaded {len(actions)} actions from {macro_file}")

            # Activate window if not in test mode
            if not self.test_mode:
                import win32gui
                win32gui.SetForegroundWindow(self.window_handle)
                time.sleep(0.5)  # Wait for window activation

            # Play actions
            last_time = None
            for action in actions:
                if last_time:
                    # Maintain original timing between actions
                    time_diff = action['timestamp'] - last_time
                    time.sleep(max(0, time_diff))

                # Convert to screen coordinates
                if self.window_rect:
                    screen_x = self.window_rect[0] + action['x']
                    screen_y = self.window_rect[1] + action['y']
                else:
                    screen_x = action['x']
                    screen_y = action['y']

                # Execute action
                if action['type'] == 'move':
                    self.direct_input.move_mouse(screen_x, screen_y)
                elif action['type'] == 'click':
                    self.direct_input.move_mouse(screen_x, screen_y)
                    time.sleep(0.1)
                    self.direct_input.click()

                last_time = action['timestamp']
                self.logger.debug(f"Executed action: {action}")

            self.logger.info("Macro playback completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during macro playback: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

def main():
    try:
        # Initialize tester with appropriate test mode
        test_mode = platform.system() != 'Windows'
        tester = MacroTester(test_mode=test_mode)

        # Find window (use "Notepad" for testing)
        window_title = "Notepad" if not test_mode else "Test Window"
        if not tester.find_window(window_title):
            print(f"Please open {window_title} for testing")
            return

        # Record some actions
        print("Starting recording in 3 seconds...")
        time.sleep(3)
        tester.start_recording()

        # Record for 10 seconds
        print("Recording mouse movements for 10 seconds...")
        recording_duration = 10
        start_time = time.time()

        while time.time() - start_time < recording_duration:
            # Get current mouse position
            if test_mode:
                # Simulate mouse movement in test mode
                x = int((time.time() - start_time) * 50) % 800
                y = 300 + int(math.sin(time.time()) * 100)
            else:
                import win32gui
                flags, hcursor, (x, y) = win32gui.GetCursorInfo()

            tester.record_action('move', x, y)
            time.sleep(0.1)  # Sample every 100ms

        # Stop recording
        tester.stop_recording()
        print("Recording stopped")

        # Play back the recorded macro
        print("Playing back recorded macro in 3 seconds...")
        time.sleep(3)
        tester.play_macro()

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()