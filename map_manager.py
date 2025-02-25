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

        # Refined arrow detection parameters for optimal accuracy
        self.arrow_color_lower = np.array([85, 115, 155])  # Further refined color range
        self.arrow_color_upper = np.array([140, 250, 255])
        self.min_arrow_area = 150  # Increased minimum area
        self.max_arrow_area = 350  # Tighter maximum area

        # Fine-tuned shape validation parameters
        self.min_arrow_aspect_ratio = 0.6   # Stricter ratio for better accuracy
        self.max_arrow_aspect_ratio = 1.5   # Tighter maximum for arrow shape
        self.min_arrow_solidity = 0.85      # Higher solidity requirement
        self.max_circularity = 0.55         # Lower circularity threshold

        # Optimized morphological operation parameters
        self.morph_kernel_size = 2  # Small kernel for detail preservation
        self.morph_iterations = 1   # Single iteration to maintain shape

        # Enhanced direction detection weights
        self.tip_weight = 0.98      # Even higher weight on tip direction
        self.rect_weight = 0.02     # Minimal rectangle weight

        # Additional angle correction parameters
        self.angle_correction_factor = 1.15  # Adjusted correction factor
        self.max_angle_deviation = 35.0     # Tighter maximum deviation

        # Cardinal direction snapping threshold (in degrees)
        self.cardinal_snap_threshold = 7.0  # Slightly increased snap threshold

        # Additional angular thresholds
        self.angle_noise_threshold = 12.0   # Tighter noise filtering
        self.tip_angle_threshold = 25.0     # Stricter tip angle deviation limit

        # Minimap to world scale factors
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

        # Updated terrain detection settings with shallow water distinction
        self.terrain_colors = {
            'deep_water': {
                'lower': np.array([95, 150, 150]),  # Darker blue for deep water
                'upper': np.array([125, 255, 255])
            },
            'shallow_water': {
                'lower': np.array([95, 60, 180]),  # Lighter blue for shallow water
                'upper': np.array([125, 150, 255])
            },
            'mountain': {
                'lower': np.array([0, 0, 40]),  # Dark terrain
                'upper': np.array([30, 40, 160])
            },
            'cliff': {
                'lower': np.array([0, 0, 20]),  # Very dark terrain
                'upper': np.array([180, 25, 80])
            }
        }

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

    def detect_player_position(self, minimap_image: np.ndarray) -> Optional[PlayerPosition]:
        """Detect player position and direction from minimap with improved accuracy"""
        try:
            if self.minimap_size == (0, 0):
                self.minimap_size = minimap_image.shape[:2]

            # Convert to HSV and create mask
            hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
            arrow_mask = cv2.inRange(hsv, self.arrow_color_lower, self.arrow_color_upper)

            # Enhanced mask cleanup
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            for _ in range(self.morph_iterations):
                arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_OPEN, kernel)
                arrow_mask = cv2.morphologyEx(arrow_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Enhanced contour filtering with scoring
            valid_contours = []
            contour_scores = []

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < self.min_arrow_area or area > self.max_arrow_area:
                    continue

                rect = cv2.minAreaRect(cnt)
                width = float(rect[1][0])
                height = float(rect[1][1])
                if width == 0 or height == 0:
                    continue

                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio < self.min_arrow_aspect_ratio or aspect_ratio > self.max_arrow_aspect_ratio:
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                if hull_area == 0:
                    continue

                solidity = area / hull_area
                if solidity < self.min_arrow_solidity:
                    continue

                perimeter = float(cv2.arcLength(cnt, True))
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > self.max_circularity:
                    continue

                # Calculate comprehensive shape score
                shape_score = float(
                    solidity * 0.4 +
                    (1 - abs(aspect_ratio - 1.2) / 1.2) * 0.3 +
                    (1 - circularity) * 0.3
                )

                # Area score based on target size
                target_area = (self.min_arrow_area + self.max_arrow_area) / 2
                area_score = float(1 - abs(area - target_area) / target_area)

                # Final weighted score
                total_score = shape_score * 0.7 + area_score * 0.3

                valid_contours.append(cnt)
                contour_scores.append(total_score)

            if not valid_contours:
                return None

            # Select best contour
            best_idx = int(np.argmax(contour_scores))
            arrow_contour = valid_contours[best_idx]

            # Calculate centroid
            M = cv2.moments(arrow_contour)
            if M["m00"] == 0:
                return None

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Improved angle calculation
            hull = cv2.convexHull(arrow_contour)
            if len(hull) >= 3:
                # Get all points and their distances from centroid
                tip_candidates = []
                for point in hull:
                    px, py = float(point[0][0]), float(point[0][1])
                    dist = float(np.sqrt((px - cx)**2 + (py - cy)**2))
                    # Calculate rough angle to filter unlikely tip points
                    rough_angle = float(np.degrees(np.arctan2(cy - py, px - cx)))
                    tip_candidates.append((dist, px, py, rough_angle))

                # Sort by distance and filter points by rough angle
                tip_candidates.sort(reverse=True, key=lambda x: x[0])

                # Select points with reasonable angles for tip calculation
                valid_angles = []
                weights = [0.6, 0.3, 0.1]  # Weight distribution for angles

                for (_, px, py, rough_angle), weight in zip(tip_candidates[:3], weights):
                    # Calculate precise angle and normalize
                    angle = float(np.degrees(np.arctan2(cy - py, px - cx)))
                    angle = float((90 - angle) % 360)  # Convert to game coordinates

                    # Filter out noisy measurements
                    if abs(angle - rough_angle) > self.angle_noise_threshold:
                        continue

                    # Apply correction factor
                    angle = angle * self.angle_correction_factor
                    valid_angles.append(angle * weight)

                tip_angle = float(sum(valid_angles))

                # Get rectangle angle as backup
                rect = cv2.minAreaRect(arrow_contour)
                rect_angle = float(-rect[-1])
                if rect[1][0] < rect[1][1]:
                    rect_angle = float((rect_angle + 90) % 360)
                else:
                    rect_angle = float(rect_angle % 360)

                # Combine angles with weights
                final_angle = float((tip_angle * self.tip_weight +
                                   rect_angle * self.rect_weight) % 360)

                # Cardinal direction snapping
                cardinal_angles = [0, 90, 180, 270]
                for cardinal in cardinal_angles:
                    if abs(final_angle - cardinal) < self.cardinal_snap_threshold:
                        final_angle = float(cardinal)
                        break

                # Check for extreme deviations
                if abs(final_angle - tip_angle) > self.max_angle_deviation:
                    final_angle = tip_angle  # Use tip angle if rect angle deviates too much
            else:
                # Fallback to rectangle angle
                rect = cv2.minAreaRect(arrow_contour)
                final_angle = float((-rect[-1]) % 360)

            # Create player position
            position = PlayerPosition(cx, cy, final_angle)
            self.last_known_position = position

            # Enhanced logging for angle detection
            self.logger.debug(
                f"Arrow detection details:\n"
                f"Position: ({cx}, {cy})\n"
                f"Raw tip angle: {tip_angle:.1f}°\n"
                f"Rectangle angle: {rect_angle:.1f}°\n"
                f"Final angle: {final_angle:.1f}°\n"
                f"Quality score: {contour_scores[best_idx]:.2f}\n"
                f"Valid angle count: {len(valid_angles)}"
            )

            return position

        except Exception as e:
            self.logger.error(f"Error detecting player position: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_player_direction_vector(self) -> Optional[Tuple[float, float]]:
        """Get normalized direction vector based on player's facing direction"""
        if not self.last_known_position:
            return None

        # Convert angle to radians
        angle_rad = np.radians(self.last_known_position.direction)

        # Calculate normalized direction vector (dx, dy)
        dx = np.sin(angle_rad)  # sin for x because 0° is North
        dy = -np.cos(angle_rad)  # -cos for y because y increases downward

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