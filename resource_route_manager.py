"""Resource route management system for automated farming"""
import logging
from typing import List, Tuple, Dict, Optional
from pathfinding import PathFinder
import numpy as np

class ResourceRoute:
    def __init__(self):
        self.waypoints: List[Tuple[int, int]] = []
        self.resource_types: List[str] = []
        self.total_distance: float = 0.0
        self.estimated_time: float = 0.0
        self.cycle_complete: bool = False
        self.resource_counts: Dict[str, int] = {}

    def add_waypoint(self, position: Tuple[int, int], resource_type: str):
        """Add a waypoint to the route"""
        self.waypoints.append(position)
        self.resource_types.append(resource_type)
        self.resource_counts[resource_type] = self.resource_counts.get(resource_type, 0) + 1

    def optimize(self):
        """Optimize route by nearest neighbor algorithm with state awareness"""
        if len(self.waypoints) <= 2:
            return

        # Start with first point
        optimized = [self.waypoints[0]]
        remaining = self.waypoints[1:]
        current = self.waypoints[0]

        # Track visited resource types to ensure variety
        visited_types = {self.resource_types[0]: 1}

        while remaining:
            # Calculate scores for each potential next point
            scores = []
            for i, point in enumerate(remaining):
                # Base score is distance (lower is better)
                distance = abs(point[0] - current[0]) + abs(point[1] - current[1])
                resource_type = self.resource_types[self.waypoints.index(point)]

                # Adjust score based on resource type variety
                type_count = visited_types.get(resource_type, 0)
                variety_bonus = 1.0 / (type_count + 1)  # Prefer less visited types

                # Final score (lower is better)
                score = distance * (1.0 - variety_bonus * 0.3)  # 30% weight to variety
                scores.append((score, i))

            # Choose point with best score
            best_score, best_idx = min(scores)
            chosen_point = remaining[best_idx]
            chosen_type = self.resource_types[self.waypoints.index(chosen_point)]

            # Update tracking
            optimized.append(chosen_point)
            remaining.pop(best_idx)
            current = chosen_point
            visited_types[chosen_type] = visited_types.get(chosen_type, 0) + 1

        # Complete the cycle if requested
        if self.cycle_complete and optimized:
            optimized.append(optimized[0])

        self.waypoints = optimized
        # Update resource types to match new waypoint order
        self.resource_types = [
            self.resource_types[self.waypoints.index(point)]
            for point in optimized
        ]

class ResourceRouteManager:
    def __init__(self, grid_size: int = 32):
        self.logger = logging.getLogger('ResourceRouteManager')
        self.pathfinder = PathFinder(grid_size)
        self.current_route: Optional[ResourceRoute] = None
        self.detected_resources: Dict[str, List[Tuple[int, int]]] = {}
        self.resource_states: Dict[Tuple[int, int], bool] = {}  # True if full/available

    def add_resource_location(self, resource_type: str, position: Tuple[int, int], is_full: bool = True):
        """Add a detected resource location with state"""
        if resource_type not in self.detected_resources:
            self.detected_resources[resource_type] = []
        if position not in self.detected_resources[resource_type]:
            self.detected_resources[resource_type].append(position)
            self.resource_states[position] = is_full
        self.logger.debug(f"Added {resource_type} at position {position} ({'full' if is_full else 'empty'})")

    def create_route(self, 
                    resource_types: List[str], 
                    start_pos: Tuple[int, int],
                    max_distance: Optional[float] = None,
                    complete_cycle: bool = True,
                    prefer_full: bool = True) -> Optional[ResourceRoute]:
        """Create an optimized route through specified resource types"""
        try:
            route = ResourceRoute()
            route.cycle_complete = complete_cycle
            route.add_waypoint(start_pos, "start")

            # Collect available resource positions
            available_positions = []
            for res_type in resource_types:
                positions = self.detected_resources.get(res_type, [])
                # Filter by state if preferring full resources
                if prefer_full:
                    positions = [pos for pos in positions if self.resource_states.get(pos, True)]
                available_positions.extend([(pos, res_type) for pos in positions])

            if not available_positions:
                self.logger.warning("No suitable resources found for specified types")
                return None

            # Filter by max distance if specified
            if max_distance is not None:
                available_positions = [
                    (pos, res_type) for pos, res_type in available_positions
                    if (abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])) <= max_distance
                ]

            # Add positions to route
            for pos, res_type in available_positions:
                route.add_waypoint(pos, res_type)

            # Optimize route
            route.optimize()

            # Calculate paths between waypoints
            full_path = []
            bounds = self._calculate_bounds(route.waypoints)

            for i in range(len(route.waypoints) - 1):
                path = self.pathfinder.find_path(
                    route.waypoints[i],
                    route.waypoints[i + 1],
                    bounds
                )
                if path:
                    full_path.extend(path)
                    route.total_distance += len(path)
                else:
                    self.logger.warning(f"Could not find path between {route.waypoints[i]} and {route.waypoints[i + 1]}")

            # Estimate time based on distance and resource types
            base_time = route.total_distance * 0.5  # 0.5 seconds per grid move
            gathering_time = len(route.waypoints) * 2.0  # 2 seconds per resource
            route.estimated_time = base_time + gathering_time

            self.current_route = route
            self.logger.info(
                f"Created route with {len(route.waypoints)} waypoints "
                f"covering {len(route.resource_counts)} resource types, "
                f"total distance: {route.total_distance:.1f}, "
                f"estimated time: {route.estimated_time:.1f}s"
            )

            return route

        except Exception as e:
            self.logger.error(f"Error creating route: {str(e)}")
            return None

    def visualize_route(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Create a visualization of the current route with resource states"""
        try:
            import cv2

            # Create blank image
            visualization = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

            if not self.current_route or not self.current_route.waypoints:
                return visualization

            # Define colors for different resource types
            type_colors = {
                'start': (255, 255, 255),  # White
                'copper_ore_full': (0, 165, 255),  # Orange
                'copper_ore_empty': (128, 128, 128),  # Gray
                'limestone_full': (255, 255, 224),  # Light yellow
                'limestone_empty': (169, 169, 169),  # Dark gray
                'ore': (139, 69, 19),  # Brown
                'stone': (128, 128, 128),  # Gray
                'fiber': (0, 255, 0),  # Green
                'hide': (210, 105, 30),  # Chocolate
            }

            # Draw connections between waypoints
            for i in range(len(self.current_route.waypoints) - 1):
                start = self.current_route.waypoints[i]
                end = self.current_route.waypoints[i + 1]

                # Scale coordinates to image size
                start_scaled = (
                    int(start[0] * image_size[0] / self.pathfinder.grid_size),
                    int(start[1] * image_size[1] / self.pathfinder.grid_size)
                )
                end_scaled = (
                    int(end[0] * image_size[0] / self.pathfinder.grid_size),
                    int(end[1] * image_size[1] / self.pathfinder.grid_size)
                )

                # Draw path line
                cv2.line(visualization, start_scaled, end_scaled, (0, 255, 0), 2)

                # Draw resource markers
                resource_type = self.current_route.resource_types[i]
                color = type_colors.get(resource_type, (0, 255, 255))

                # Draw larger circle for start point
                if resource_type == 'start':
                    cv2.circle(visualization, start_scaled, 8, color, -1)
                else:
                    # Draw resource node with state-dependent appearance
                    is_full = self.resource_states.get(start, True)
                    if is_full:
                        cv2.circle(visualization, start_scaled, 6, color, -1)
                    else:
                        cv2.circle(visualization, start_scaled, 6, color, 2)

            # Draw final point
            final_type = self.current_route.resource_types[-1]
            final_color = type_colors.get(final_type, (0, 255, 255))
            final_pos = self.current_route.waypoints[-1]
            final_scaled = (
                int(final_pos[0] * image_size[0] / self.pathfinder.grid_size),
                int(final_pos[1] * image_size[1] / self.pathfinder.grid_size)
            )
            cv2.circle(visualization, final_scaled, 6, final_color, -1)

            return visualization

        except Exception as e:
            self.logger.error(f"Error visualizing route: {str(e)}")
            return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    def _calculate_bounds(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate map bounds based on positions"""
        if not positions:
            return (100, 100)  # Default size

        x_coords, y_coords = zip(*positions)
        max_x = max(x_coords) + 50  # Add margin
        max_y = max(y_coords) + 50
        return (max_x, max_y)