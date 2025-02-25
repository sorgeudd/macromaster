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

        # Arrow detection parameters from gameplay footage
        self.arrow_color_ranges = [
            # Yellow arrow
            (np.array([20, 100, 200]), np.array([30, 255, 255])),
            # White arrow
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Green arrow
            (np.array([45, 100, 100]), np.array([75, 255, 255]))
        ]

    def load_map(self, map_name: str) -> bool:
        """Load a map and its resource data"""
        try:
            # Load map image for terrain analysis
            map_path = self.maps_directory / f"{map_name}.png"
            if not map_path.exists():
                self.logger.error(f"Map file not found: {map_path}")
                return False

            self.current_map = cv2.imread(str(map_path))
            if self.current_map is None:
                self.logger.error(f"Failed to load map: {map_path}")
                return False

            # Load resource data
            resource_path = self.maps_directory / f"{map_name}_resources.json"
            if resource_path.exists():
                import json
                with open(resource_path) as f:
                    resource_data = json.load(f)
                    self.resource_nodes[map_name] = [
                        ResourceNode(
                            position=(node['x'], node['y']),
                            resource_type=node['type'],
                            priority=node.get('priority', 1.0)
                        )
                        for node in resource_data.get('nodes', [])
                    ]
                    self.logger.info(f"Loaded {len(self.resource_nodes[map_name])} resource nodes")
            else:
                self.logger.warning(f"No resource data found for map: {map_name}")
                self.resource_nodes[map_name] = []

            self.current_map_name = map_name
            self.logger.info(f"Loaded map: {map_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading map: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def detect_player_position(self, game_frame: np.ndarray) -> Optional[PlayerPosition]:
        """Detect player position from game window frame"""
        try:
            if self.minimap_size == (0, 0):
                self.minimap_size = game_frame.shape[:2]
                self.logger.debug(f"Set minimap size to {self.minimap_size}")

            # Convert to HSV for more robust detection
            hsv = cv2.cvtColor(game_frame, cv2.COLOR_BGR2HSV)

            # Create combined mask for all arrow colors
            arrow_mask = np.zeros(game_frame.shape[:2], dtype=np.uint8)
            for lower, upper in self.arrow_color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                arrow_mask = cv2.bitwise_or(arrow_mask, mask)

            # Clean up mask
            kernel = np.ones((2, 2), np.uint8)
            arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_OPEN, kernel)
            arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                self.logger.debug("No arrow contours found")
                return None

            # Find arrow-like shape
            best_contour = None
            best_score = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10 or area > 500:  # Filter by area
                    continue

                # Check shape properties
                perimeter = cv2.arcLength(cnt, True)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)

                if hull_area == 0:
                    continue

                solidity = area / hull_area
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Arrow should have medium solidity and low circularity
                if 0.4 <= solidity <= 0.8 and circularity < 0.6:
                    rect = cv2.minAreaRect(cnt)
                    width, height = rect[1]
                    if width == 0 or height == 0:
                        continue

                    aspect_ratio = max(width, height) / min(width, height)
                    if 1.5 <= aspect_ratio <= 4.0:  # Arrow should be elongated
                        score = solidity * (1 - circularity) * min(aspect_ratio / 4.0, 1.0)
                        if score > best_score:
                            best_score = score
                            best_contour = cnt

            if best_contour is None:
                self.logger.debug("No suitable arrow shape found")
                return None

            # Find arrow direction using tip point
            M = cv2.moments(best_contour)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Find tip point (furthest point from centroid)
            tip_point = None
            max_dist = 0
            for point in best_contour:
                dist = np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2)
                if dist > max_dist:
                    max_dist = dist
                    tip_point = point[0]

            if tip_point is None:
                return None

            # Calculate angle from centroid to tip
            dx = tip_point[0] - cx
            dy = cy - tip_point[1]  # Invert y since image coordinates increase downward
            angle = np.degrees(np.arctan2(dx, dy)) % 360

            # Create debug visualization
            debug_frame = cv2.cvtColor(arrow_mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 1)
            cv2.circle(debug_frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.circle(debug_frame, tuple(tip_point), 3, (255, 0, 0), -1)
            cv2.line(debug_frame, (cx, cy), tuple(tip_point), (255, 255, 0), 1)
            cv2.imwrite('arrow_detection.png', debug_frame)

            position = PlayerPosition(cx, cy, angle)
            self.last_known_position = position
            self.logger.debug(f"Detected player at ({cx}, {cy}) facing {angle:.1f}°")
            return position

        except Exception as e:
            self.logger.error(f"Error detecting player position: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

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

            # Simple color-based terrain classification
            # These thresholds should be calibrated based on your map colors
            if color[0] > 200:  # Blue channel
                return 'water'
            elif color[1] > 200:  # Green channel
                return 'forest'
            elif np.mean(color) < 50:  # Dark areas
                return 'mountain'
            else:
                return 'normal'

        except Exception as e:
            self.logger.error(f"Error getting terrain type: {str(e)}")
            return 'unknown'

    def get_terrain_penalty(self, position: Tuple[int, int]) -> float:
        """Get movement penalty for terrain type"""
        terrain_type = self.get_terrain_type(position)

        # Define movement penalties for different terrain
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