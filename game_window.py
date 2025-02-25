"""Game window movement and coordinate translation system"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List

class GameWindow:
    # Supported resolutions as class constants
    SUPPORTED_RESOLUTIONS = [
        (1280, 720),   # HD
        (1366, 768),   # HD+
        (1600, 900),   # HD+ Widescreen
        (1920, 1080),  # Full HD
        (2560, 1440),  # QHD
        (3440, 1440),  # Ultrawide
        (3840, 2160)   # 4K UHD
    ]

    MIN_SYSTEM_WIDTH = 1024
    MIN_SYSTEM_HEIGHT = 768
    MAX_SYSTEM_WIDTH = 7680  # Support up to 8K
    MAX_SYSTEM_HEIGHT = 4320

    def __init__(self):
        self.logger = logging.getLogger('GameWindow')

        # System resolution (default 1920x1200)
        self.system_width = 1920
        self.system_height = 1200

        # Game window parameters (default 1280x720)
        self.window_width = 1280
        self.window_height = 720
        self.window_x = (self.system_width - self.window_width) // 2
        self.window_y = (self.system_height - self.window_height) // 2

        # Game view parameters
        self.view_width = self.window_width
        self.view_height = self.window_height
        self.center_x = self.view_width // 2
        self.center_y = self.view_height // 2

        # Camera parameters
        self.camera_rotation = 0.0  # Current camera rotation in degrees
        self.zoom_level = 1.0  # Current zoom level

        # Movement settings
        self.movement_speed = 100  # Pixels per second at default zoom
        self.rotation_speed = 45  # Degrees per second

        # Viewport parameters
        self.viewport_offset_x = 0  # Offset from center of game window
        self.viewport_offset_y = 0

    @classmethod
    def get_supported_resolutions(cls) -> List[Tuple[int, int]]:
        """Get list of supported game window resolutions"""
        return cls.SUPPORTED_RESOLUTIONS

    @classmethod
    def get_system_resolution_bounds(cls) -> Dict[str, int]:
        """Get minimum and maximum supported system resolutions"""
        return {
            'min_width': cls.MIN_SYSTEM_WIDTH,
            'min_height': cls.MIN_SYSTEM_HEIGHT,
            'max_width': cls.MAX_SYSTEM_WIDTH,
            'max_height': cls.MAX_SYSTEM_HEIGHT
        }

    def is_resolution_supported(self, width: int, height: int) -> bool:
        """Check if a given resolution is supported"""
        return (width, height) in self.SUPPORTED_RESOLUTIONS

    def get_closest_supported_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the closest supported resolution to the given dimensions"""
        if (width, height) in self.SUPPORTED_RESOLUTIONS:
            return (width, height)

        closest = min(self.SUPPORTED_RESOLUTIONS, 
                     key=lambda x: abs(x[0] - width) + abs(x[1] - height))
        return closest

    def configure_resolution(self, game_width: int = None, game_height: int = None, 
                           system_width: int = None, system_height: int = None) -> bool:
        """Configure window and system resolutions with validation"""
        try:
            # Update system resolution if provided
            if system_width and system_height:
                # Strict resolution validation
                if system_width < self.MIN_SYSTEM_WIDTH or system_height < self.MIN_SYSTEM_HEIGHT:
                    self.logger.error(f"System resolution {system_width}x{system_height} too small, "
                                    f"minimum is {self.MIN_SYSTEM_WIDTH}x{self.MIN_SYSTEM_HEIGHT}")
                    return False

                if system_width > self.MAX_SYSTEM_WIDTH or system_height > self.MAX_SYSTEM_HEIGHT:
                    self.logger.error(f"System resolution {system_width}x{system_height} too large, "
                                    f"maximum is {self.MAX_SYSTEM_WIDTH}x{self.MAX_SYSTEM_HEIGHT}")
                    return False

                # Validate aspect ratio
                aspect_ratio = system_width / system_height
                if aspect_ratio < 1.3 or aspect_ratio > 2.35:
                    self.logger.warning(f"Unusual aspect ratio {aspect_ratio:.2f}, might cause display issues. "
                                      f"Recommended range: 1.3 - 2.35")

                self.system_width = system_width
                self.system_height = system_height
                self.logger.info(f"Updated system resolution to {self.system_width}x{self.system_height}")

            # Update game window resolution if provided
            if game_width and game_height:
                # Find closest supported resolution
                if not self.is_resolution_supported(game_width, game_height):
                    closest = self.get_closest_supported_resolution(game_width, game_height)
                    self.logger.warning(f"Unsupported resolution {game_width}x{game_height}. "
                                      f"Using closest supported resolution {closest[0]}x{closest[1]}")
                    game_width, game_height = closest

                # Update game window parameters
                self.window_width = game_width
                self.window_height = game_height
                self.view_width = game_width
                self.view_height = game_height
                self.center_x = self.view_width // 2
                self.center_y = self.view_height // 2

                # Ensure window fits within system bounds
                if self.window_width > self.system_width or self.window_height > self.system_height:
                    self.logger.warning(f"Window size {self.window_width}x{self.window_height} larger than "
                                      f"system resolution, adjusting...")

                    # Maintain aspect ratio while scaling down
                    scale = min(self.system_width / self.window_width,
                              self.system_height / self.window_height)

                    self.window_width = int(self.window_width * scale)
                    self.window_height = int(self.window_height * scale)
                    self.view_width = self.window_width
                    self.view_height = self.window_height

                    # Recenter after scaling
                    self.window_x = (self.system_width - self.window_width) // 2
                    self.window_y = (self.system_height - self.window_height) // 2

                # Reset viewport offset when resolution changes
                self.viewport_offset_x = 0
                self.viewport_offset_y = 0

                self.logger.info(f"Updated game window resolution to {self.window_width}x{self.window_height}")
                self.logger.debug(f"Window position: ({self.window_x}, {self.window_y})")

            return True

        except Exception as e:
            self.logger.error(f"Error configuring resolution: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def set_viewport_offset(self, offset_x: int, offset_y: int):
        """Set viewport offset with bounds checking"""
        try:
            # Calculate maximum allowed offsets
            max_offset_x = self.window_width - self.view_width
            max_offset_y = self.window_height - self.view_height

            # Clamp offsets within bounds
            self.viewport_offset_x = max(-max_offset_x, min(offset_x, max_offset_x))
            self.viewport_offset_y = max(-max_offset_y, min(offset_y, max_offset_y))

            self.logger.debug(f"Set viewport offset to ({self.viewport_offset_x}, {self.viewport_offset_y})")
            return True
        except Exception as e:
            self.logger.error(f"Error setting viewport offset: {str(e)}")
            return False

    def translate_screen_to_window(self, screen_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert system screen coordinates to game window coordinates"""
        try:
            # Adjust for window position
            window_x = screen_pos[0] - self.window_x
            window_y = screen_pos[1] - self.window_y

            # Account for viewport offset
            window_x -= self.viewport_offset_x
            window_y -= self.viewport_offset_y

            # Ensure coordinates are within window bounds
            window_x = max(0, min(window_x, self.window_width - 1))
            window_y = max(0, min(window_y, self.window_height - 1))

            self.logger.debug(f"Translated screen pos {screen_pos} to window pos ({window_x}, {window_y})")
            return (int(window_x), int(window_y))

        except Exception as e:
            self.logger.error(f"Error translating screen coordinates: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return center of window as fallback
            return (self.window_width // 2, self.window_height // 2)

    def translate_window_to_screen(self, window_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert game window coordinates to system screen coordinates"""
        try:
            # Account for viewport offset
            screen_x = window_pos[0] + self.viewport_offset_x
            screen_y = window_pos[1] + self.viewport_offset_y

            # Add window position offset
            screen_x += self.window_x
            screen_y += self.window_y

            # Ensure coordinates are within screen bounds
            screen_x = max(0, min(screen_x, self.system_width - 1))
            screen_y = max(0, min(screen_y, self.system_height - 1))

            self.logger.debug(f"Translated window pos {window_pos} to screen pos ({screen_x}, {screen_y})")
            return (int(screen_x), int(screen_y))

        except Exception as e:
            self.logger.error(f"Error translating window coordinates: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return center of screen as fallback
            return (self.system_width // 2, self.system_height // 2)

    def translate_world_to_screen(self, world_pos: Tuple[float, float], player_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates with error handling"""
        try:
            # Calculate relative position from player
            rel_x = world_pos[0] - player_pos[0]
            rel_y = world_pos[1] - player_pos[1]

            # Apply camera rotation
            angle_rad = np.radians(self.camera_rotation)
            rot_x = rel_x * np.cos(angle_rad) - rel_y * np.sin(angle_rad)
            rot_y = rel_x * np.sin(angle_rad) + rel_y * np.cos(angle_rad)

            # Apply zoom and center offset
            screen_x = int(self.center_x + rot_x * self.zoom_level)
            screen_y = int(self.center_y + rot_y * self.zoom_level)

            # Account for viewport offset
            screen_x += self.viewport_offset_x
            screen_y += self.viewport_offset_y

            # Ensure coordinates are within bounds
            screen_x = max(0, min(screen_x, self.window_width - 1))
            screen_y = max(0, min(screen_y, self.window_height - 1))

            return (screen_x, screen_y)

        except Exception as e:
            self.logger.error(f"Error translating world coordinates: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return center of screen as fallback
            return (self.center_x, self.center_y)

    def get_movement_vector(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate normalized movement vector towards target"""
        try:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]

            # Normalize vector
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length

            return (dx, dy)
        except Exception as e:
            self.logger.error(f"Error calculating movement vector: {str(e)}")
            return (0.0, 0.0)  # Return zero vector as fallback

    def calculate_screen_movement(self, 
                                  current_pos: Tuple[float, float],
                                  target_pos: Tuple[float, float],
                                  delta_time: float) -> Tuple[float, float]:
        """Calculate screen-space movement towards target position"""
        # Get movement direction
        dx, dy = self.get_movement_vector(current_pos, target_pos)

        # Apply movement speed and time delta
        move_x = dx * self.movement_speed * delta_time
        move_y = dy * self.movement_speed * delta_time

        # Convert to screen coordinates with zoom
        screen_x = move_x * self.zoom_level
        screen_y = move_y * self.zoom_level

        # Account for viewport offset
        screen_x += self.viewport_offset_x
        screen_y += self.viewport_offset_y

        return (screen_x, screen_y)

    def is_position_visible(self, world_pos: Tuple[float, float], player_pos: Tuple[float, float]) -> bool:
        """Check if a world position is visible in the game window"""
        screen_pos = self.translate_world_to_screen(world_pos, player_pos)

        # Account for viewport offset
        screen_x = screen_pos[0] - self.viewport_offset_x
        screen_y = screen_pos[1] - self.viewport_offset_y

        return (0 <= screen_x < self.view_width and 
                0 <= screen_y < self.view_height)

    def get_screen_direction(self, angle: float) -> Tuple[float, float]:
        """Convert a world angle to screen-space direction vector"""
        # Adjust angle for screen space (0 is right, 90 is down)
        screen_angle = angle - 90 - self.camera_rotation
        angle_rad = np.radians(screen_angle)

        # Get normalized direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        return (dx, dy)

    def update_view_size(self, view_width: int = None, view_height: int = None):
        """Update view size and recalculate viewport bounds"""
        try:
            if view_width is not None:
                self.view_width = min(view_width, self.window_width)
            if view_height is not None:
                self.view_height = min(view_height, self.window_height)

            # Recenter viewport if needed
            self.center_x = self.view_width // 2
            self.center_y = self.view_height // 2

            # Re-clamp viewport offsets with new bounds
            self.set_viewport_offset(self.viewport_offset_x, self.viewport_offset_y)

            self.logger.debug(f"Updated view size to {self.view_width}x{self.view_height}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating view size: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def update_screen_metrics(self, system_width: int = None, system_height: int = None):
        """Update system screen metrics and recalculate window position"""
        try:
            if system_width and system_height:
                # Validate resolution
                if system_width < self.MIN_SYSTEM_WIDTH or system_height < self.MIN_SYSTEM_HEIGHT:
                    self.logger.error(f"System resolution {system_width}x{system_height} too small, minimum is {self.MIN_SYSTEM_WIDTH}x{self.MIN_SYSTEM_HEIGHT}")
                    return False

                if system_width > self.MAX_SYSTEM_WIDTH or system_height > self.MAX_SYSTEM_HEIGHT:
                    self.logger.error(f"System resolution {system_width}x{system_height} too large, maximum is {self.MAX_SYSTEM_WIDTH}x{self.MAX_SYSTEM_HEIGHT}")
                    return False

                # Update metrics
                self.system_width = system_width
                self.system_height = system_height

                # Recalculate window position to keep it centered
                self.window_x = (self.system_width - self.window_width) // 2
                self.window_y = (self.system_height - self.window_height) // 2

                # Ensure window fits within system bounds
                if self.window_width > self.system_width or self.window_height > self.system_height:
                    self.logger.warning(f"Window size {self.window_width}x{self.window_height} larger than system resolution, adjusting...")

                    # Maintain aspect ratio while scaling down
                    scale = min(self.system_width / self.window_width,
                              self.system_height / self.window_height)

                    self.window_width = int(self.window_width * scale)
                    self.window_height = int(self.window_height * scale)
                    self.view_width = self.window_width
                    self.view_height = self.window_height

                    # Recenter after scaling
                    self.window_x = (self.system_width - self.window_width) // 2
                    self.window_y = (self.system_height - self.window_height) // 2

                self.logger.info(f"Updated system metrics to {system_width}x{system_height}")
                self.logger.debug(f"New window position: ({self.window_x}, {self.window_y})")
                self.logger.debug(f"New window size: {self.window_width}x{self.window_height}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error updating screen metrics: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False