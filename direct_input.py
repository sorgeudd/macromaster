"""Windows-compatible DirectInput implementation for smooth mouse control"""
import ctypes
import time
import math
import random
import traceback
from ctypes import Structure, c_long, c_ulong, sizeof, POINTER, pointer
import logging
import platform

# Windows API Constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class MOUSEINPUT(Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_ulong),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", POINTER(c_ulong))
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(Structure):
    _fields_ = [
        ("type", c_ulong),
        ("union", INPUT_UNION)
    ]

class DirectInput:
    def __init__(self, test_mode=False):
        self.logger = logging.getLogger('DirectInput')
        self.test_mode = test_mode
        self.mock_cursor_pos = (0, 0)  # For test mode
        self.mock_button_state = {
            'left': False,
            'right': False,
            'middle': False
        }

        try:
            if platform.system() == 'Windows' and not test_mode:
                try:
                    self.user32 = ctypes.windll.user32
                    # Get virtual screen metrics
                    self.screen_left = self.user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
                    self.screen_top = self.user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
                    self.screen_width = self.user32.GetSystemMetrics(78) # SM_CXVIRTUALSCREEN
                    self.screen_height = self.user32.GetSystemMetrics(79)# SM_CYVIRTUALSCREEN
                    self.logger.info(f"Initialized DirectInput with virtual screen: {self.screen_width}x{self.screen_height}")
                except AttributeError:
                    raise ImportError("Windows-specific functionality not available")
            else:
                # Use default values for test mode
                self.screen_left = 0
                self.screen_top = 0
                self.screen_width = 1920 if not test_mode else 800
                self.screen_height = 1080 if not test_mode else 600
                self.logger.info("Running in test mode with default screen metrics")
                self.user32 = None
        except Exception as e:
            self.logger.error(f"Failed to initialize DirectInput: {e}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            # Use test mode defaults
            self.screen_left = 0
            self.screen_top = 0
            self.screen_width = 800
            self.screen_height = 600
            self.user32 = None
            self.logger.info("Falling back to test mode")

    def _normalize_coordinates(self, x, y):
        """Convert screen coordinates to normalized coordinates (0-65535 range)"""
        try:
            # Adjust coordinates relative to virtual screen boundaries
            x = x - self.screen_left
            y = y - self.screen_top

            # Ensure coordinates are within virtual screen bounds
            x = max(0, min(x, self.screen_width))
            y = max(0, min(y, self.screen_height))

            # Convert to normalized range (0-65535)
            norm_x = int((x * 65535) / self.screen_width)
            norm_y = int((y * 65535) / self.screen_height)

            return norm_x, norm_y
        except Exception as e:
            self.logger.error(f"Error normalizing coordinates: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return 0, 0

    def move_mouse(self, x, y, smooth=True, speed=1.0):
        """Move mouse using SendInput with natural movement simulation"""
        try:
            if self.test_mode:
                self.logger.info(f"Test mode: Moving mouse to ({x}, {y})")
                self.mock_cursor_pos = (x, y)
                return True

            # Get current position
            current = POINT()
            self.user32.GetCursorPos(pointer(current))

            # Calculate path points
            if smooth:
                # Calculate number of steps based on distance and speed
                distance = math.sqrt((x - current.x)**2 + (y - current.y)**2)
                steps = int(max(10, min(50, distance / 10)))  # Dynamic step count

                # Add subtle randomization for more natural movement
                path_points = []
                for i in range(steps + 1):
                    t = i / steps
                    # Use bezier curve for smooth acceleration/deceleration
                    smooth_t = t * t * (3 - 2 * t)
                    point_x = current.x + (x - current.x) * smooth_t
                    point_y = current.y + (y - current.y) * smooth_t

                    # Add slight randomization
                    if i > 0 and i < steps:
                        rand_x = random.uniform(-2, 2)
                        rand_y = random.uniform(-2, 2)
                        point_x += rand_x
                        point_y += rand_y

                    path_points.append((int(point_x), int(point_y)))

                # Move through path points
                for px, py in path_points:
                    self._send_mouse_input(px, py)
                    delay = random.uniform(0.001, 0.003) / speed
                    time.sleep(delay)
            else:
                # Direct movement
                self._send_mouse_input(x, y)

            return True

        except Exception as e:
            self.logger.error(f"Error moving mouse: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def _send_mouse_input(self, x, y):
        """Send a single mouse movement input"""
        try:
            if self.test_mode:
                self.logger.debug(f"Test mode: Sending mouse input to ({x}, {y})")
                self.mock_cursor_pos = (x, y)
                return True

            # Convert to normalized coordinates
            norm_x, norm_y = self._normalize_coordinates(x, y)

            # Prepare input structure
            mouse_input = MOUSEINPUT()
            mouse_input.dx = norm_x
            mouse_input.dy = norm_y
            mouse_input.mouseData = 0
            mouse_input.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
            mouse_input.time = 0
            mouse_input.dwExtraInfo = pointer(c_ulong(0))

            input_struct = INPUT()
            input_struct.type = 0  # INPUT_MOUSE
            input_struct.union.mi = mouse_input

            # Send input
            result = self.user32.SendInput(1, pointer(input_struct), sizeof(INPUT))
            if result != 1:
                self.logger.error("SendInput failed")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error sending mouse input: {str(e)}")
            return False

    def click(self, button='left', double=False, hold_duration=0.1):
        """Perform mouse click using SendInput"""
        try:
            if self.test_mode:
                self.logger.info(f"Test mode: {'Double ' if double else ''}Clicking {button} button")
                self.mock_button_state[button] = not self.mock_button_state[button]
                return True

            # Select appropriate button flags
            if button == 'left':
                down_flag = MOUSEEVENTF_LEFTDOWN
                up_flag = MOUSEEVENTF_LEFTUP
            elif button == 'right':
                down_flag = MOUSEEVENTF_RIGHTDOWN
                up_flag = MOUSEEVENTF_RIGHTUP
            elif button == 'middle':
                down_flag = MOUSEEVENTF_MIDDLEDOWN
                up_flag = MOUSEEVENTF_MIDDLEUP
            else:
                self.logger.error(f"Invalid button type: {button}")
                return False

            def send_click():
                # Send button down
                mouse_input = MOUSEINPUT()
                mouse_input.dx = 0
                mouse_input.dy = 0
                mouse_input.mouseData = 0
                mouse_input.dwFlags = down_flag
                mouse_input.time = 0
                mouse_input.dwExtraInfo = pointer(c_ulong(0))

                input_struct = INPUT()
                input_struct.type = 0  # INPUT_MOUSE
                input_struct.union.mi = mouse_input

                self.user32.SendInput(1, pointer(input_struct), sizeof(INPUT))
                time.sleep(hold_duration)  # Hold duration

                # Send button up
                mouse_input.dwFlags = up_flag
                input_struct.union.mi = mouse_input
                self.user32.SendInput(1, pointer(input_struct), sizeof(INPUT))

            # Perform click(s)
            send_click()
            if double:
                time.sleep(0.1)  # Delay between clicks for double-click
                send_click()

            return True
        except Exception as e:
            self.logger.error(f"Error performing click: {str(e)}")
            return False

    def drag(self, start_x, start_y, end_x, end_y, button='left', speed=1.0):
        """Perform drag operation"""
        try:
            # Move to start position
            self.move_mouse(start_x, start_y, smooth=True, speed=speed)
            time.sleep(0.1)

            # Press button
            self.click(button=button, hold_duration=0.0)  # Just press, don't release
            time.sleep(0.1)

            # Move to end position
            self.move_mouse(end_x, end_y, smooth=True, speed=speed)
            time.sleep(0.1)

            # Release button
            self.click(button=button)  # Normal click to release
            return True

        except Exception as e:
            self.logger.error(f"Error performing drag: {str(e)}")
            return False

    def get_cursor_pos(self):
        """Get current cursor position"""
        if self.test_mode:
            return self.mock_cursor_pos

        try:
            point = POINT()
            self.user32.GetCursorPos(pointer(point))
            return (point.x, point.y)
        except Exception as e:
            self.logger.error(f"Error getting cursor position: {e}")
            return (0, 0)

    def scroll(self, clicks, horizontal=False):
        """Scroll the mouse wheel"""
        try:
            if self.test_mode:
                self.logger.info(f"Test mode: Scrolling {'horizontally' if horizontal else 'vertically'} by {clicks}")
                return True

            mouse_input = MOUSEINPUT()
            mouse_input.dx = 0
            mouse_input.dy = 0
            mouse_input.mouseData = clicks * 120  # 120 units per click is standard
            mouse_input.dwFlags = MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL
            mouse_input.time = 0
            mouse_input.dwExtraInfo = pointer(c_ulong(0))

            input_struct = INPUT()
            input_struct.type = 0  # INPUT_MOUSE
            input_struct.union.mi = mouse_input

            result = self.user32.SendInput(1, pointer(input_struct), sizeof(INPUT))
            return result == 1

        except Exception as e:
            self.logger.error(f"Error scrolling: {str(e)}")
            return False