"""Game window movement and coordinate translation system"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

class GameWindow:
    def __init__(self):
        self.logger = logging.getLogger('GameWindow')
        
        # Game window parameters
        self.view_width = 1920  # Default game window width
        self.view_height = 1080  # Default game window height
        self.center_x = self.view_width // 2
        self.center_y = self.view_height // 2
        
        # Camera parameters
        self.camera_rotation = 0.0  # Current camera rotation
        self.zoom_level = 1.0  # Current zoom level
        
        # Movement settings
        self.movement_speed = 100  # Pixels per second at default zoom
        self.rotation_speed = 45  # Degrees per second
        
    def translate_world_to_screen(self, world_pos: Tuple[float, float], player_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
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
        
        return (screen_x, screen_y)
    
    def get_movement_vector(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate normalized movement vector towards target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Normalize vector
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
            
        return (dx, dy)
    
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
        
        # Convert to screen coordinates
        screen_x = move_x * self.zoom_level
        screen_y = move_y * self.zoom_level
        
        return (screen_x, screen_y)
    
    def is_position_visible(self, world_pos: Tuple[float, float], player_pos: Tuple[float, float]) -> bool:
        """Check if a world position is visible in the game window"""
        screen_pos = self.translate_world_to_screen(world_pos, player_pos)
        
        return (0 <= screen_pos[0] < self.view_width and 
                0 <= screen_pos[1] < self.view_height)
    
    def get_screen_direction(self, angle: float) -> Tuple[float, float]:
        """Convert a world angle to screen-space direction vector"""
        # Adjust angle for screen space (0 is right, 90 is down)
        screen_angle = angle - 90 - self.camera_rotation
        angle_rad = np.radians(screen_angle)
        
        # Get normalized direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        return (dx, dy)
