"""Test script for macro recording and playback functionality"""
import logging
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import platform
import traceback
import math
import random
from direct_input import DirectInput

class MacroAction:
    """Class to represent a macro action with validation"""
    def __init__(self, action_type: str, x: Optional[int] = None, y: Optional[int] = None, 
                 button: str = 'left', duration: float = 0.1, scroll_amount: int = 0):
        self.type = action_type
        self.x = x
        self.y = y
        self.button = button
        self.duration = duration
        self.scroll_amount = scroll_amount
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        """Convert action to dictionary for JSON serialization"""
        return {
            'type': self.type,
            'x': self.x,
            'y': self.y,
            'button': self.button,
            'duration': self.duration,
            'scroll_amount': self.scroll_amount,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MacroAction':
        """Create action from dictionary"""
        action = cls(
            action_type=data['type'],
            x=data.get('x'),
            y=data.get('y'),
            button=data.get('button', 'left'),
            duration=data.get('duration', 0.1),
            scroll_amount=data.get('scroll_amount', 0)
        )
        action.timestamp = data['timestamp']
        return action

    def validate(self) -> bool:
        """Validate action data"""
        if self.type not in ['move', 'click', 'drag', 'scroll']:
            return False
        if self.type in ['move', 'drag'] and (self.x is None or self.y is None):
            return False
        if self.type == 'click' and self.button not in ['left', 'right', 'middle']:
            return False
        return True

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
        self.recorded_actions: List[MacroAction] = []
        self.window_handle = None
        self.window_rect = None
        self.last_mouse_pos = (0, 0)
        self.drag_start = None

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
        self.last_mouse_pos = (0, 0)
        self.drag_start = None
        self.logger.info("Started recording mouse movements")
        return True

    def record_action(self, action_type: str, x: Optional[int] = None, y: Optional[int] = None,
                     button: str = 'left', duration: float = 0.1, scroll_amount: int = 0) -> None:
        """Record a mouse action"""
        if not self.recording:
            return

        try:
            # Convert to window-relative coordinates
            if x is not None and y is not None:
                if self.window_rect:
                    x = x - self.window_rect[0]
                    y = y - self.window_rect[1]

                # Check if this is potentially the end of a drag operation
                if self.drag_start is not None:
                    dx = x - self.drag_start[0]
                    dy = y - self.drag_start[1]
                    if math.sqrt(dx*dx + dy*dy) > 5:  # Minimum drag distance
                        action = MacroAction('drag', self.drag_start[0], self.drag_start[1],
                                          button=button, duration=duration)
                        if action.validate():
                            self.recorded_actions.append(action)
                            self.logger.debug(f"Recorded drag start: {action.to_dict()}")

            action = MacroAction(action_type, x, y, button, duration, scroll_amount)
            if action.validate():
                self.recorded_actions.append(action)
                self.logger.debug(f"Recorded action: {action.to_dict()}")

                # Update state
                if x is not None and y is not None:
                    self.last_mouse_pos = (x, y)
                if action_type == 'click':
                    self.drag_start = (x, y)
                elif action_type != 'drag':
                    self.drag_start = None

        except Exception as e:
            self.logger.error(f"Error recording action: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")

    def stop_recording(self) -> bool:
        """Stop recording and save actions"""
        if not self.recording:
            return False

        try:
            self.recording = False
            self.drag_start = None

            # Save recorded actions
            actions_data = [action.to_dict() for action in self.recorded_actions]
            with open('recorded_macro.json', 'w') as f:
                json.dump(actions_data, f, indent=2)
            self.logger.info(f"Saved {len(self.recorded_actions)} actions to recorded_macro.json")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save recorded actions: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def play_macro(self, macro_file: str = 'recorded_macro.json', speed: float = 1.0) -> bool:
        """Play back recorded macro"""
        if not self.window_handle and not self.test_mode:
            self.logger.error("No window selected for playback")
            return False

        try:
            # Load macro file
            with open(macro_file, 'r') as f:
                actions_data = json.load(f)

            actions = [MacroAction.from_dict(data) for data in actions_data]
            self.logger.info(f"Loaded {len(actions)} actions from {macro_file}")

            # Validate all actions before playback
            for action in actions:
                if not action.validate():
                    self.logger.error(f"Invalid action in macro: {action.to_dict()}")
                    return False

            # Activate window if not in test mode
            if not self.test_mode:
                import win32gui
                win32gui.SetForegroundWindow(self.window_handle)
                time.sleep(0.5)  # Wait for window activation

            # Play actions
            last_time = None
            for action in actions:
                if last_time:
                    # Maintain original timing between actions, adjusted by speed
                    time_diff = (action.timestamp - last_time) / speed
                    time.sleep(max(0, time_diff))

                # Convert to screen coordinates if needed
                screen_x = None
                screen_y = None
                if action.x is not None and action.y is not None:
                    if self.window_rect:
                        screen_x = self.window_rect[0] + action.x
                        screen_y = self.window_rect[1] + action.y
                    else:
                        screen_x = action.x
                        screen_y = action.y

                # Execute action
                try:
                    if action.type == 'move':
                        self.direct_input.move_mouse(screen_x, screen_y, smooth=True, speed=speed)
                    elif action.type == 'click':
                        if screen_x is not None and screen_y is not None:
                            self.direct_input.move_mouse(screen_x, screen_y, smooth=True, speed=speed)
                            time.sleep(0.1 / speed)
                        self.direct_input.click(button=action.button, hold_duration=action.duration)
                    elif action.type == 'drag':
                        if 'end_x' in action.to_dict() and 'end_y' in action.to_dict():
                            end_x = self.window_rect[0] + action.to_dict()['end_x'] if self.window_rect else action.to_dict()['end_x']
                            end_y = self.window_rect[1] + action.to_dict()['end_y'] if self.window_rect else action.to_dict()['end_y']
                            self.direct_input.drag(screen_x, screen_y, end_x, end_y, button=action.button, speed=speed)
                    elif action.type == 'scroll':
                        self.direct_input.scroll(action.scroll_amount)
                except Exception as e:
                    self.logger.error(f"Error executing action {action.to_dict()}: {e}")
                    continue

                last_time = action.timestamp
                self.logger.debug(f"Executed action: {action.to_dict()}")

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
                t = time.time() - start_time
                x = int((math.sin(t) * 0.5 + 0.5) * 800)
                y = int((math.cos(t * 0.5) * 0.5 + 0.5) * 600)

                # Add some random clicks and scrolls
                if random.random() < 0.05:
                    tester.record_action('click', x, y)
                elif random.random() < 0.05:
                    tester.record_action('scroll', scroll_amount=random.randint(-3, 3))
                else:
                    tester.record_action('move', x, y)
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
        tester.play_macro(speed=1.0)  # Normal speed playback

    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()