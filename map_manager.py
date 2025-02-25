"""Map management system for terrain-aware navigation"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, NamedTuple

class PlayerPosition(NamedTuple):
    x: int
    y: int
    direction: float  # in degrees, 0 is North, 90 is East, etc.

class MapManager:
    def __init__(self, maps_directory: str = "maps"):
        self.logger = logging.getLogger('MapManager')
        self.maps_directory = Path(maps_directory)
        self.maps_directory.mkdir(exist_ok=True)

        # Map cache
        self.loaded_maps: Dict[str, np.ndarray] = {}
        self.current_map: Optional[np.ndarray] = None
        self.current_map_name: str = ""

        # Player tracking
        self.last_known_position: Optional[PlayerPosition] = None
        self.minimap_size: Tuple[int, int] = (0, 0)  # Will be set when processing minimap

        # Arrow detection parameters based on game footage analysis
        self.arrow_color_lower = np.array([90, 160, 180])  # Verified from footage
        self.arrow_color_upper = np.array([110, 200, 255])
        self.min_arrow_area = 2     # From footage: min area was 2.0
        self.max_arrow_area = 5     # From footage: max area was 4.5

        # Shape parameters based on footage analysis
        self.min_arrow_aspect_ratio = 1.0   # From footage: width/height varied
        self.max_arrow_aspect_ratio = 2.0   # Allow for perspective changes
        self.min_arrow_solidity = 0.6      # Relaxed for small pixel count
        self.max_circularity = 0.8         # Adjusted for small size

        # Morphological operation parameters
        self.morph_kernel_size = 2   # Minimal kernel for small arrow
        self.morph_iterations = 1    # Single iteration to preserve shape

        # Direction detection weights
        self.tip_weight = 0.7       # Emphasis on tip for direction
        self.rect_weight = 0.3      # Less weight on rectangle due to small size

        # Angular parameters from footage analysis
        self.angle_correction_factor = 1.0  # No correction needed
        self.max_angle_deviation = 45.0     # From footage: saw up to ~45° changes

        # Cardinal direction snapping
        self.cardinal_snap_threshold = 10.0  # More permissive for small arrow

        # Additional thresholds
        self.angle_noise_threshold = 30.0   # Higher for small pixel movements
        self.tip_angle_threshold = 45.0     # From footage analysis

        # Scale factors
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

        # Terrain colors from game footage
        self.terrain_colors = {
            'deep_water': {
                'lower': np.array([95, 150, 150]),
                'upper': np.array([125, 255, 255])
            },
            'shallow_water': {
                'lower': np.array([95, 60, 180]),
                'upper': np.array([125, 150, 255])
            },
            'mountain': {
                'lower': np.array([0, 0, 40]),
                'upper': np.array([30, 40, 160])
            },
            'cliff': {
                'lower': np.array([0, 0, 20]),
                'upper': np.array([180, 25, 80])
            }
        }

    def detect_player_position(self, minimap_image: np.ndarray) -> Optional[PlayerPosition]:
        """Detect player position and direction from minimap"""
        try:
            if self.minimap_size == (0, 0):
                self.minimap_size = minimap_image.shape[:2]

            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
            arrow_mask = cv2.inRange(hsv, self.arrow_color_lower, self.arrow_color_upper)

            # Find contours without morphological operations to preserve tiny shapes
            contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Filter contours by area
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_arrow_area <= area <= self.max_arrow_area:
                    valid_contours.append(cnt)
                    self.logger.debug(f"Found valid contour with area: {area}")

            if not valid_contours:
                return None

            # Use largest valid contour
            arrow_contour = max(valid_contours, key=cv2.contourArea)

            # Get centroid
            M = cv2.moments(arrow_contour)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get convex hull for more reliable tip detection
            hull = cv2.convexHull(arrow_contour)

            # Find the tip point (furthest point from centroid)
            max_dist = 0
            tip_point = None

            for point in hull:
                pt = point[0]
                dist = np.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                if dist > max_dist:
                    max_dist = dist
                    tip_point = pt

            if tip_point is None:
                self.logger.error("Failed to find tip point")
                return None

            # Calculate angle in game coordinates (0° is North, clockwise)
            dx = tip_point[0] - cx
            dy = cy - tip_point[1]  # Reverse y because image coordinates increase downward

            # Calculate raw angle (arctan2 gives -π to π)
            raw_angle = np.degrees(np.arctan2(dx, dy))
            # Convert to game angles (0° at North, clockwise)
            game_angle = raw_angle % 360

            # Snap to cardinal directions if close
            for cardinal in [0, 90, 180, 270]:
                if abs(game_angle - cardinal) < self.cardinal_snap_threshold:
                    game_angle = float(cardinal)
                    self.logger.debug(f"Snapped to cardinal angle: {game_angle}°")
                    break

            # Debug logging
            self.logger.debug(
                f"Arrow detection details:\n"
                f"Centroid: ({cx}, {cy})\n"
                f"Tip point: ({tip_point[0]}, {tip_point[1]})\n"
                f"dx, dy: ({dx}, {dy})\n"
                f"Raw angle: {raw_angle}°\n"
                f"Game angle: {game_angle}°"
            )

            # Create position object
            position = PlayerPosition(cx, cy, game_angle)
            self.last_known_position = position

            return position

        except Exception as e:
            self.logger.error(f"Error detecting player position: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def load_map(self, map_name: str, map_image: Optional[np.ndarray] = None) -> bool:
        """Load a map by name or from an image"""
        try:
            if map_image is not None:
                # Update scale factors for coordinate translation
                self.scale_x = map_image.shape[1] / self.minimap_size[1] if self.minimap_size[1] > 0 else 1.0
                self.scale_y = map_image.shape[0] / self.minimap_size[0] if self.minimap_size[0] > 0 else 1.0

                # Preprocess image for better terrain detection
                processed_map = self._preprocess_map(map_image)
                self.loaded_maps[map_name] = processed_map
                self.current_map = processed_map
                self.current_map_name = map_name
                return True

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

            processed_map = self._preprocess_map(map_image)
            self.loaded_maps[map_name] = processed_map
            self.current_map = processed_map
            self.current_map_name = map_name

            self.logger.info(f"Successfully loaded map: {map_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading map {map_name}: {str(e)}")
            return False

    def minimap_to_world_coords(self, minimap_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert minimap coordinates to world map coordinates"""
        world_x = int(minimap_pos[0] * self.scale_x)
        world_y = int(minimap_pos[1] * self.scale_y)

        # Ensure coordinates are within world map bounds
        if self.current_map is not None:
            world_x = max(0, min(world_x, self.current_map.shape[1] - 1))
            world_y = max(0, min(world_y, self.current_map.shape[0] - 1))

        return (world_x, world_y)

    def world_to_minimap_coords(self, world_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world map coordinates to minimap coordinates"""
        minimap_x = int(world_pos[0] / self.scale_x)
        minimap_y = int(world_pos[1] / self.scale_y)

        # Ensure coordinates are within minimap bounds
        if self.minimap_size != (0, 0):
            minimap_x = max(0, min(minimap_x, self.minimap_size[1] - 1))
            minimap_y = max(0, min(minimap_y, self.minimap_size[0] - 1))

        return (minimap_x, minimap_y)


    def get_player_direction_vector(self) -> Optional[Tuple[float, float]]:
        """Get normalized direction vector based on player's facing direction"""
        if not self.last_known_position:
            return None

        # Convert angle to radians (adjusted for game's coordinate system)
        angle_rad = np.radians(self.last_known_position.direction)

        # Calculate direction vector
        dx = np.sin(angle_rad)  # sin for x because 0° is North
        dy = -np.cos(angle_rad) # -cos for y because y increases downward

        # Normalize vector
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        return (dx, dy)

    def detect_terrain(self, position: Tuple[int, int], use_world_coords: bool = False) -> Dict[str, float]:
        """Detect terrain features at given position"""
        if self.current_map is None:
            return {'deep_water': 0.0, 'shallow_water': 0.0, 'mountain': 0.0, 'cliff': 0.0}

        try:
            # Convert world coordinates to map coordinates if needed
            if use_world_coords:
                pos = self.world_to_minimap_coords(position)
            else:
                pos = position

            # Get color at position and surrounding area
            radius = 3  # Analysis radius
            x_start = max(0, pos[0] - radius)
            x_end = min(self.current_map.shape[1], pos[0] + radius + 1)
            y_start = max(0, pos[1] - radius)
            y_end = min(self.current_map.shape[0], pos[1] + radius + 1)

            area = self.current_map[y_start:y_end, x_start:x_end]

            terrain_scores = {}
            for terrain_type, color_range in self.terrain_colors.items():
                # Check if colors in area are within range
                mask = cv2.inRange(area, color_range['lower'], color_range['upper'])
                score = np.mean(mask) / 255.0
                # Apply smoothing to reduce noise
                terrain_scores[terrain_type] = min(score * 1.2, 1.0)

            return terrain_scores

        except Exception as e:
            self.logger.error(f"Error detecting terrain at {position}: {str(e)}")
            return {'deep_water': 0.0, 'shallow_water': 0.0, 'mountain': 0.0, 'cliff': 0.0}

    def get_terrain_penalty(self, position: Tuple[int, int], use_world_coords: bool = False) -> float:
        """Calculate terrain penalty for pathfinding"""
        terrain_scores = self.detect_terrain(position, use_world_coords)

        # Deep water is impassable
        if terrain_scores['deep_water'] > 0.3:
            return 1.0

        # Shallow water has a moderate movement penalty
        if terrain_scores['shallow_water'] > 0.3:
            return 0.4  # 40% movement penalty in shallow water

        # Mountains and cliffs increase movement cost
        mountain_penalty = terrain_scores['mountain'] * 0.6  # Reduced mountain penalty
        cliff_penalty = terrain_scores['cliff'] * 0.8      # Cliff penalty

        # Combined penalty with emphasis on cliffs
        penalty = max(mountain_penalty, cliff_penalty)

        return min(penalty, 0.9)  # Cap at 90% to allow movement

    def _preprocess_map(self, map_image: np.ndarray) -> np.ndarray:
        """Preprocess map for better terrain detection"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)

        # Apply light contrast enhancement
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        return hsv

    def extract_map_name_from_minimap(self, minimap_image: np.ndarray) -> Optional[str]:
        """Extract map name from game minimap using template matching"""
        try:
            # Compare with known map templates
            known_maps = {
                "forest": "forest_template.png",
                "mountain": "mountain_template.png",
                "swamp": "swamp_template.png"
            }

            best_match = None
            best_score = 0.0

            for map_name, template_file in known_maps.items():
                template_path = self.maps_directory / template_file
                if template_path.exists():
                    template = cv2.imread(str(template_path))
                    if template is not None:
                        # Template matching
                        result = cv2.matchTemplate(minimap_image, template, cv2.TM_CCOEFF_NORMED)
                        _, score, _, _ = cv2.minMaxLoc(result)

                        if score > best_score:
                            best_score = score
                            best_match = map_name

            if best_match and best_score > 0.7:  # 70% confidence threshold
                return best_match

            return "default_map"

        except Exception as e:
            self.logger.error(f"Error extracting map name: {str(e)}")
            return None