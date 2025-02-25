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
        self.current_zone: str = ""

        # Scale factors for coordinate translation
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

        # Arrow detection parameters for gameplay window
        self.arrow_color_ranges = [
            # Yellow arrow
            (np.array([20, 100, 200]), np.array([30, 255, 255])),
            # White arrow
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Green arrow
            (np.array([45, 100, 100]), np.array([75, 255, 255]))
        ]

    def get_zone_name(self, minimap_image: np.ndarray) -> Optional[str]:
        """Extract zone name from minimap image
        This would need to be implemented based on how zone names appear in the minimap
        For now, return None to indicate no zone change detected
        """
        # TODO: Implement OCR or template matching to detect zone name
        return None

    def update_from_minimap(self, minimap_image: np.ndarray, resource_type: str = None) -> bool:
        """Update state from minimap image, including zone detection"""
        try:
            # Detect player position first
            position = self.detect_player_position(minimap_image)
            if position:
                self.last_known_position = position

            # Try to detect zone name
            zone_name = self.get_zone_name(minimap_image)
            if zone_name and zone_name != self.current_zone:
                self.logger.info(f"Zone changed to: {zone_name}")
                self.current_zone = zone_name

                # If resource type specified, load appropriate map
                if resource_type:
                    map_name = f"{zone_name.lower()} {resource_type}"
                    return self.load_map(map_name)

            return True

        except Exception as e:
            self.logger.error(f"Error updating from minimap: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def detect_resource_spots(self, map_image: np.ndarray, resource_type: str) -> List[ResourceNode]:
        """Detect resource spots from map image"""
        try:
            spots = []
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)

            # Create mask for blue markers (adjusted for actual map markers)
            blue_lower = np.array([90, 100, 100])  # Relaxed blue HSV thresholds
            blue_upper = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, blue_lower, blue_upper)

            # Clean up mask
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Save debug visualizations
            debug_path = self.maps_directory / f"{self.current_map_name}_mask.png"
            cv2.imwrite(str(debug_path), mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Filter by area for spot markers
                area = cv2.contourArea(cnt)
                if area < 5 or area > 100:  # Adjust based on spot size
                    continue

                # Calculate centroid
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Create resource node
                    spots.append(ResourceNode(
                        position=(cx, cy),
                        resource_type=resource_type,
                        priority=1.0
                    ))
                    self.logger.debug(f"Added {resource_type} spot at ({cx}, {cy})")

            # Save visualization
            debug_viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for spot in spots:
                cv2.circle(debug_viz, spot.position, 5, (0, 255, 0), -1)
            viz_path = self.maps_directory / f"{self.current_map_name}_spots.png"
            cv2.imwrite(str(viz_path), debug_viz)

            self.logger.debug(f"Detected {len(spots)} {resource_type} spots")
            return spots

        except Exception as e:
            self.logger.error(f"Error detecting resource spots: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return []

    def load_map(self, map_name: str) -> bool:
        """Load a map by name (e.g., 'mase knoll fish', 'mase knoll ore')"""
        try:
            # Clean up map name for file paths
            clean_name = map_name.lower().replace(' ', '_')
            map_path = self.maps_directory / f"{clean_name}.png"

            if not map_path.exists():
                self.logger.error(f"Map file not found: {map_path}")
                return False

            # Load map image
            self.current_map = cv2.imread(str(map_path))
            if self.current_map is None:
                self.logger.error(f"Failed to load map: {map_path}")
                return False

            # Get resource type from map name
            resource_type = map_name.split()[-1]  # Last word is resource type

            # Detect resource locations
            spots = self.detect_resource_spots(self.current_map, resource_type)
            self.resource_nodes[clean_name] = spots

            self.current_map_name = clean_name
            self.logger.info(f"Loaded {len(spots)} {resource_type} locations from map: {map_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading map: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def detect_player_position(self, minimap_image: np.ndarray) -> Optional[PlayerPosition]:
        """Detect player position and direction from game window minimap"""
        try:
            if self.minimap_size == (0, 0):
                self.minimap_size = minimap_image.shape[:2]
                self.logger.debug(f"Set minimap size to {self.minimap_size}")

            # Convert to HSV for more robust detection
            hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)

            # Create combined mask for all arrow colors
            arrow_mask = np.zeros(minimap_image.shape[:2], dtype=np.uint8)
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

        self.logger.debug(f"Found {len(nearby)} resources within {radius} units")
        return nearby

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