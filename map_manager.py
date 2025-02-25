"""Map management system for resource tracking and navigation"""
import cv2
import numpy as np
import logging
import traceback
from pathlib import Path
from typing import Dict, Tuple, Optional, List, NamedTuple
from dataclasses import dataclass

class ResourceNode(NamedTuple):
    position: Tuple[int, int]
    resource_type: str
    priority: float = 1.0

@dataclass
class PlayerPosition:
    """Represents the current player position"""
    x: int
    y: int

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

        # Arrow detection parameters
        self.arrow_color_ranges = [
            # Yellow arrow in HSV
            (np.array([25, 100, 150]), np.array([35, 255, 255])),
            # White arrow
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Green arrow
            (np.array([45, 100, 100]), np.array([75, 255, 255]))
        ]

    def get_zone_name(self, minimap_image: np.ndarray) -> Optional[str]:
        """Extract zone name from minimap image using OCR"""
        try:
            # Isolate the top portion of the minimap where zone name usually appears
            height, width = minimap_image.shape[:2]
            top_portion = minimap_image[0:int(height*0.1), :]  # Top 10% of image

            # Create a mask for white/light text
            hsv = cv2.cvtColor(top_portion, cv2.COLOR_BGR2HSV)
            # Detect white/light colored text
            lower = np.array([0, 0, 200])
            upper = np.array([180, 30, 255])
            text_mask = cv2.inRange(hsv, lower, upper)

            # Save debug image
            debug_path = self.maps_directory / "zone_text_mask.png"
            cv2.imwrite(str(debug_path), text_mask)

            # For now, return a fixed zone name for testing
            # In a real implementation, this would use OCR to read the text
            # TODO: Implement actual OCR using tesseract or similar
            return "Mase Knoll"

        except Exception as e:
            self.logger.error(f"Error detecting zone name: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
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
        """Detect player position from minimap"""
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
            contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                self.logger.debug("No arrow contours found")
                return None

            # Find best arrow-like shape using simple area/perimeter ratio
            best_contour = None
            best_score = 0
            best_centroid = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10 or area > 100:  # Adjusted for test arrow size
                    continue

                # Get centroid
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Basic shape scoring
                perimeter = cv2.arcLength(cnt, True)
                shape_score = area / (perimeter * perimeter)  # Compactness ratio

                if shape_score > best_score:
                    best_score = shape_score
                    best_contour = cnt
                    best_centroid = (cx, cy)

            if best_contour is None:
                self.logger.debug("No suitable arrow shape found")
                return None

            # Save debug visualization
            debug_frame = minimap_image.copy()
            cv2.drawContours(debug_frame, [best_contour], -1, (0, 255, 0), 1)
            cv2.circle(debug_frame, best_centroid, 3, (0, 0, 255), -1)
            cv2.imwrite('arrow_detection_debug.png', debug_frame)

            # Create position using centroid coordinates
            position = PlayerPosition(best_centroid[0], best_centroid[1])
            self.last_known_position = position
            self.logger.debug(f"Detected player at ({position.x}, {position.y})")
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