"""Map management system for resource tracking and navigation"""
import cv2
import numpy as np
import logging
import traceback
from pathlib import Path
from typing import Dict, Tuple, Optional, NamedTuple, List

class ResourceNode(NamedTuple):
    position: Tuple[int, int]
    resource_type: str
    priority: float = 1.0

class PlayerPosition(NamedTuple):
    x: int
    y: int
    direction: float  # in degrees, 0 is North, 90 is East, etc.

class MapManager:
    def __init__(self, maps_directory: str = "maps"):
        self.logger = logging.getLogger('MapManager')
        self.logger.setLevel(logging.DEBUG)
        self.maps_directory = Path(maps_directory)
        self.maps_directory.mkdir(exist_ok=True)

        # Resource tracking
        self.resource_nodes: Dict[str, List[ResourceNode]] = {}
        self.active_resources: List[ResourceNode] = []

        # Current state
        self.current_map: Optional[np.ndarray] = None
        self.current_map_name: str = ""
        self.last_known_position: Optional[PlayerPosition] = None
        self.minimap_size: Tuple[int, int] = (0, 0)

        # Scale factors for coordinate translation
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

        # Fishing spot detection thresholds (from real map analysis)
        self.fishing_spot_thresholds = {
            'blue_marker': {
                'lower': np.array([15, 50, 150]),  # HSV thresholds for spots
                'upper': np.array([35, 150, 255])
            }
        }

    def detect_fishing_spots(self, map_image: np.ndarray) -> List[ResourceNode]:
        """Detect fishing spots from map image"""
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)

            # Create mask for spots
            mask = cv2.inRange(
                hsv,
                self.fishing_spot_thresholds['blue_marker']['lower'],
                self.fishing_spot_thresholds['blue_marker']['upper']
            )

            # Clean up mask
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Save debug mask
            cv2.imwrite('fishing_spots_mask.png', mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fishing_spots = []
            for cnt in contours:
                # Filter by area
                area = cv2.contourArea(cnt)
                if area < 5 or area > 100:  # Adjust based on spot size
                    continue

                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get color at position
                    color = map_image[cy, cx]
                    self.logger.debug(f"Spot color at ({cx}, {cy}): BGR={color}")

                    # Verify spot is in valid position
                    if self.get_terrain_type((cx, cy)) == 'water':
                        fishing_spots.append(ResourceNode(
                            position=(cx, cy),
                            resource_type="fishing_spot",
                            priority=1.0
                        ))
                        self.logger.debug(f"Added fishing spot at ({cx}, {cy})")
                    else:
                        self.logger.debug(f"Skipped non-water spot at ({cx}, {cy})")

            # Create debug visualization
            debug_viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for spot in fishing_spots:
                cv2.circle(debug_viz, spot.position, 5, (0, 255, 0), -1)
            cv2.imwrite('fishing_spots_debug.png', debug_viz)

            self.logger.debug(f"Detected {len(fishing_spots)} valid fishing spots")
            return fishing_spots

        except Exception as e:
            self.logger.error(f"Error detecting fishing spots: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return []

    def load_map(self, map_name: str) -> bool:
        """Load a map and detect its resources"""
        try:
            # Load map image for resource detection
            map_path = self.maps_directory / f"{map_name}.png"
            if not map_path.exists():
                self.logger.error(f"Map file not found: {map_path}")
                return False

            self.current_map = cv2.imread(str(map_path))
            if self.current_map is None:
                self.logger.error(f"Failed to load map: {map_path}")
                return False

            # Detect fishing spots from map image
            fishing_spots = self.detect_fishing_spots(self.current_map)
            self.resource_nodes[map_name] = fishing_spots
            self.logger.info(f"Detected {len(fishing_spots)} fishing spots in map")

            self.current_map_name = map_name
            self.logger.info(f"Loaded map: {map_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading map: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def get_nearby_resources(self, position: Tuple[int, int], radius: float) -> List[ResourceNode]:
        """Get resources within radius of position"""
        if not self.current_map_name or self.current_map_name not in self.resource_nodes:
            return []

        nearby = []
        for node in self.resource_nodes[self.current_map_name]:
            dx = node.position[0] - position[0]
            dy = node.position[1] - position[1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance <= radius:
                nearby.append(node)

        self.logger.debug(f"Found {len(nearby)} resources within {radius} units")
        return nearby

    def get_terrain_type(self, position: Tuple[int, int]) -> str:
        """Get terrain type at position"""
        if self.current_map is None:
            return 'unknown'

        try:
            # Ensure position is within bounds
            x = min(max(position[0], 0), self.current_map.shape[1] - 1)
            y = min(max(position[1], 0), self.current_map.shape[0] - 1)

            # Get color at position
            color = self.current_map[y, x]
            self.logger.debug(f"Color at {position}: BGR={color}")

            # From actual map analysis: water areas are medium-high BGR values
            if (85 <= color[0] <= 100 and     # Blue: medium-high
                140 <= color[1] <= 160 and    # Green: high
                135 <= color[2] <= 150):      # Red: medium-high
                return 'water'
            elif color[1] > 120 and color[2] < 100:  # Strong green
                return 'forest'
            elif np.mean(color) < 60:  # Dark areas
                return 'mountain'
            else:
                return 'normal'

        except Exception as e:
            self.logger.error(f"Error getting terrain type: {str(e)}")
            return 'unknown'

    def get_terrain_penalty(self, position: Tuple[int, int]) -> float:
        """Get movement penalty for terrain type"""
        terrain_type = self.get_terrain_type(position)

        penalties = {
            'normal': 0.0,
            'water': 0.7,
            'forest': 0.3,
            'mountain': 0.8,
            'unknown': 0.5
        }

        return penalties.get(terrain_type, 0.5)

    def minimap_to_world_coords(self, minimap_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert minimap coordinates to world map coordinates"""
        world_x = int(minimap_pos[0] * self.scale_x)
        world_y = int(minimap_pos[1] * self.scale_y)
        return (world_x, world_y)

    def world_to_minimap_coords(self, world_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world map coordinates to minimap coordinates"""
        minimap_x = int(world_pos[0] / self.scale_x)
        minimap_y = int(world_pos[1] / self.scale_y)
        return (minimap_x, minimap_y)

    def get_player_direction_vector(self) -> Optional[Tuple[float, float]]:
        """Get normalized direction vector based on player's facing direction"""
        if not self.last_known_position:
            return None

        angle_rad = np.radians(self.last_known_position.direction)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)  # Negative because y increases downward
        return (dx, dy)