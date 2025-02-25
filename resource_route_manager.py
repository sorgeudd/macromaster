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

    def add_waypoint(self, position: Tuple[int, int], resource_type: str):
        """Add a waypoint to the route"""
        self.waypoints.append(position)
        self.resource_types.append(resource_type)

    def optimize(self):
        """Optimize route by nearest neighbor algorithm"""
        if len(self.waypoints) <= 2:
            return

        optimized = [self.waypoints[0]]  # Start with first point
        remaining = self.waypoints[1:]
        current = self.waypoints[0]

        while remaining:
            # Find nearest point
            nearest = min(remaining, key=lambda p: 
                        abs(p[0] - current[0]) + abs(p[1] - current[1]))
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest

        self.waypoints = optimized

class ResourceRouteManager:
    def __init__(self, grid_size: int = 32):
        self.logger = logging.getLogger('ResourceRouteManager')
        self.pathfinder = PathFinder(grid_size)
        self.current_route: Optional[ResourceRoute] = None
        self.detected_resources: Dict[str, List[Tuple[int, int]]] = {}

    def add_resource_location(self, resource_type: str, position: Tuple[int, int]):
        """Add a detected resource location"""
        if resource_type not in self.detected_resources:
            self.detected_resources[resource_type] = []
        self.detected_resources[resource_type].append(position)
        self.logger.debug(f"Added {resource_type} at position {position}")

    def clear_resources(self):
        """Clear all detected resources"""
        self.detected_resources.clear()
        self.current_route = None

    def create_route(self, resource_types: List[str], start_pos: Tuple[int, int], 
                    max_distance: Optional[float] = None) -> Optional[ResourceRoute]:
        """Create an optimized route through specified resource types"""
        try:
            route = ResourceRoute()
            route.add_waypoint(start_pos, "start")

            # Collect all relevant resource positions
            all_positions = []
            for res_type in resource_types:
                positions = self.detected_resources.get(res_type, [])
                all_positions.extend([(pos, res_type) for pos in positions])

            if not all_positions:
                self.logger.warning("No resources found for specified types")
                return None

            # Filter by max distance if specified
            if max_distance is not None:
                all_positions = [
                    (pos, res_type) for pos, res_type in all_positions
                    if (abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])) <= max_distance
                ]

            # Add positions to route
            for pos, res_type in all_positions:
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

            # Estimate time based on distance
            route.estimated_time = route.total_distance * 0.5  # 0.5 seconds per grid move

            self.current_route = route
            self.logger.info(f"Created route with {len(route.waypoints)} waypoints, "
                           f"total distance: {route.total_distance:.1f}, "
                           f"estimated time: {route.estimated_time:.1f}s")

            return route

        except Exception as e:
            self.logger.error(f"Error creating route: {str(e)}")
            return None

    def _calculate_bounds(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate map bounds based on positions"""
        if not positions:
            return (100, 100)  # Default size
        
        x_coords, y_coords = zip(*positions)
        max_x = max(x_coords) + 50  # Add margin
        max_y = max(y_coords) + 50
        return (max_x, max_y)

    def visualize_route(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Create a visualization of the current route"""
        try:
            import cv2

            # Create blank image
            visualization = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

            if not self.current_route or not self.current_route.waypoints:
                return visualization

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

                # Draw line between points
                cv2.line(visualization, start_scaled, end_scaled, (0, 255, 0), 2)

                # Draw waypoint markers
                cv2.circle(visualization, start_scaled, 5, (0, 0, 255), -1)
                cv2.circle(visualization, end_scaled, 5, (0, 0, 255), -1)

            return visualization

        except Exception as e:
            self.logger.error(f"Error visualizing route: {str(e)}")
            return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
