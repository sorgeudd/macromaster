"""Map management system for terrain-aware navigation"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

class MapManager:
    def __init__(self, maps_directory: str = "maps"):
        self.logger = logging.getLogger('MapManager')
        self.maps_directory = Path(maps_directory)
        self.maps_directory.mkdir(exist_ok=True)
        
        # Map cache
        self.loaded_maps: Dict[str, np.ndarray] = {}
        self.current_map: Optional[np.ndarray] = None
        self.current_map_name: str = ""
        
        # Terrain detection settings
        self.water_color = (64, 128, 255)  # Blue-ish color for water
        self.mountain_color = (128, 128, 128)  # Gray for mountains
        self.cliff_color = (96, 96, 96)  # Dark gray for cliffs
        
        # Initialize color ranges for terrain detection
        self.terrain_colors = {
            'water': {'lower': np.array([100, 50, 200]), 'upper': np.array([140, 255, 255])},
            'mountain': {'lower': np.array([0, 0, 100]), 'upper': np.array([180, 30, 180])},
            'cliff': {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 30, 150])}
        }
        
    def load_map(self, map_name: str) -> bool:
        """Load a map by name from the maps directory"""
        try:
            if map_name in self.loaded_maps:
                self.current_map = self.loaded_maps[map_name]
                self.current_map_name = map_name
                return True
                
            map_path = self.maps_directory / f"{map_name}.png"
            if not map_path.exists():
                self.logger.error(f"Map file not found: {map_path}")
                return False
                
            map_image = cv2.imread(str(map_path))
            if map_image is None:
                self.logger.error(f"Failed to load map image: {map_path}")
                return False
                
            self.loaded_maps[map_name] = map_image
            self.current_map = map_image
            self.current_map_name = map_name
            
            self.logger.info(f"Successfully loaded map: {map_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading map {map_name}: {str(e)}")
            return False
            
    def detect_terrain(self, position: Tuple[int, int], scale_factor: float = 1.0) -> Dict[str, float]:
        """Detect terrain features at given position"""
        if self.current_map is None:
            return {'water': 0.0, 'mountain': 0.0, 'cliff': 0.0}
            
        try:
            # Scale position to map coordinates
            x = int(position[0] * scale_factor)
            y = int(position[1] * scale_factor)
            
            if not (0 <= x < self.current_map.shape[1] and 0 <= y < self.current_map.shape[0]):
                return {'water': 0.0, 'mountain': 0.0, 'cliff': 0.0}
                
            # Get color at position
            color = self.current_map[y, x]
            hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            terrain_scores = {}
            for terrain_type, color_range in self.terrain_colors.items():
                # Check if color is within range
                in_range = cv2.inRange(
                    np.uint8([[hsv]]), 
                    color_range['lower'], 
                    color_range['upper']
                )
                terrain_scores[terrain_type] = float(in_range[0][0]) / 255.0
                
            return terrain_scores
            
        except Exception as e:
            self.logger.error(f"Error detecting terrain at {position}: {str(e)}")
            return {'water': 0.0, 'mountain': 0.0, 'cliff': 0.0}
            
    def get_terrain_penalty(self, position: Tuple[int, int], scale_factor: float = 1.0) -> float:
        """Calculate terrain penalty for pathfinding"""
        terrain_scores = self.detect_terrain(position, scale_factor)
        
        # Water is impassable
        if terrain_scores['water'] > 0.5:
            return 1.0
            
        # Mountains and cliffs increase movement cost
        penalty = max(
            terrain_scores['mountain'] * 0.8,  # 80% penalty for mountains
            terrain_scores['cliff'] * 0.9      # 90% penalty for cliffs
        )
        
        return min(penalty, 0.95)  # Cap penalty at 95% to allow some movement
        
    def extract_map_name_from_minimap(self, minimap_image: np.ndarray) -> Optional[str]:
        """Extract map name from game minimap using OCR"""
        try:
            # TODO: Implement OCR to extract map name from minimap
            # For now, return a default map name
            return "default_map"
            
        except Exception as e:
            self.logger.error(f"Error extracting map name: {str(e)}")
            return None
