"""Resource route management system for automated farming"""
import logging
from typing import List, Tuple, Dict, Optional, Set
from pathfinding import PathFinder
import numpy as np
from time import time

class ResourceTypeFilter:
    def __init__(self):
        self.enabled_types: Set[str] = set()  # Empty means all types enabled
        self.priority_modifiers: Dict[str, float] = {}  # Type-specific priority adjustments
        self.minimum_priority: float = 0.0
        self.require_full_only: bool = False
        self.cooldown_threshold: float = 300.0  # 5 minutes default

    def is_type_enabled(self, resource_type: str) -> bool:
        """Check if a resource type is enabled"""
        return not self.enabled_types or resource_type in self.enabled_types

    def get_priority_modifier(self, resource_type: str) -> float:
        """Get priority modifier for a resource type"""
        return self.priority_modifiers.get(resource_type, 1.0)

class ResourceNode:
    def __init__(self, position: Tuple[int, int], resource_type: str, is_full: bool = True):
        self.position = position
        self.resource_type = resource_type
        self.is_full = is_full
        self.last_visited = 0.0
        self.visit_count = 0
        self.cooldown = 300.0
        self.priority = 1.0
        self.filtered_out = False  # New flag for filtering

    def is_available(self, filter_settings: Optional[ResourceTypeFilter] = None) -> bool:
        """Check if node is available based on cooldown and filter settings"""
        if filter_settings:
            if not filter_settings.is_type_enabled(self.resource_type):
                return False
            if filter_settings.require_full_only and not self.is_full:
                return False
            if (time() - self.last_visited) < filter_settings.cooldown_threshold:
                return False
            if self.priority < filter_settings.minimum_priority:
                return False
        return self.is_full and (time() - self.last_visited) >= self.cooldown

    def get_effective_priority(self, filter_settings: Optional[ResourceTypeFilter] = None) -> float:
        """Calculate effective priority considering filter settings"""
        if not filter_settings:
            return self.priority
        return self.priority * filter_settings.get_priority_modifier(self.resource_type)

    def visit(self):
        """Mark node as visited"""
        self.last_visited = time()
        self.visit_count += 1

class ResourceRoute:
    def __init__(self):
        self.waypoints: List[Tuple[int, int]] = []
        self.resource_types: List[str] = []
        self.total_distance: float = 0.0
        self.estimated_time: float = 0.0
        self.cycle_complete: bool = False
        self.resource_counts: Dict[str, int] = {}
        self.nodes: List[ResourceNode] = []  # Track actual nodes for state
        self.terrain_penalties: Dict[Tuple[int, int], float] = {}  # Areas to avoid

    def add_waypoint(self, node: ResourceNode):
        """Add a waypoint using a ResourceNode"""
        self.waypoints.append(node.position)
        self.resource_types.append(node.resource_type)
        self.nodes.append(node)
        self.resource_counts[node.resource_type] = self.resource_counts.get(node.resource_type, 0) + 1

    def optimize(self, distance_weight: float = 0.6, variety_weight: float = 0.2, priority_weight: float = 0.2):
        """Optimize route considering multiple factors"""
        if len(self.waypoints) <= 2:
            return

        # Start with first point
        optimized_waypoints = [self.waypoints[0]]
        optimized_types = [self.resource_types[0]]
        optimized_nodes = [self.nodes[0]]

        remaining_indices = list(range(1, len(self.waypoints)))
        current = self.waypoints[0]

        # Track visited resource types
        visited_types = {self.resource_types[0]: 1}

        while remaining_indices:
            scores = []
            for i in remaining_indices:
                node = self.nodes[i]
                pos = node.position

                # Distance score (lower is better)
                distance = abs(pos[0] - current[0]) + abs(pos[1] - current[1])
                distance_score = 1.0 - min(distance / 100.0, 1.0)  # Normalize to 0-1

                # Variety score (higher is better)
                type_count = visited_types.get(node.resource_type, 0)
                variety_score = 1.0 / (type_count + 1)

                # Priority and availability score
                priority_score = node.priority #using effective priority here would require changes in ResourceRouteManager
                #Consider terrain penalties
                terrain_penalty = self.terrain_penalties.get(pos, 0.0)

                # Combined score (higher is better)
                total_score = (
                    distance_score * distance_weight +
                    variety_score * variety_weight +
                    priority_score * priority_weight
                ) * (1.0 - terrain_penalty)

                scores.append((total_score, i))

            # Choose point with best score
            best_score, best_idx = max(scores)
            chosen_idx = remaining_indices[best_idx]
            chosen_node = self.nodes[chosen_idx]

            # Update tracking
            optimized_waypoints.append(chosen_node.position)
            optimized_types.append(chosen_node.resource_type)
            optimized_nodes.append(chosen_node)
            visited_types[chosen_node.resource_type] = visited_types.get(chosen_node.resource_type, 0) + 1
            current = chosen_node.position

            # Remove processed index
            remaining_indices.remove(chosen_idx)

        # Complete the cycle if requested
        if self.cycle_complete and optimized_waypoints:
            optimized_waypoints.append(optimized_waypoints[0])
            optimized_types.append(optimized_types[0])
            optimized_nodes.append(optimized_nodes[0])

        self.waypoints = optimized_waypoints
        self.resource_types = optimized_types
        self.nodes = optimized_nodes

class ResourceRouteManager:
    def __init__(self, grid_size: int = 32):
        self.logger = logging.getLogger('ResourceRouteManager')
        self.pathfinder = PathFinder(grid_size)
        self.current_route: Optional[ResourceRoute] = None
        self.nodes: Dict[Tuple[int, int], ResourceNode] = {}
        self.stats: Dict[str, Dict] = {
            'total_routes': 0,
            'resources_collected': {},
            'average_completion_time': 0.0,
            'failed_routes': 0,
            'filtered_nodes': 0  # Track how many nodes were filtered out
        }
        self.filter_settings = ResourceTypeFilter()

    def configure_filtering(self,
                          enabled_types: Optional[List[str]] = None,
                          priority_modifiers: Optional[Dict[str, float]] = None,
                          minimum_priority: float = 0.0,
                          require_full_only: bool = False,
                          cooldown_threshold: Optional[float] = None):
        """Configure resource filtering settings"""
        if enabled_types is not None:
            self.filter_settings.enabled_types = set(enabled_types)
        if priority_modifiers is not None:
            self.filter_settings.priority_modifiers = priority_modifiers
        if cooldown_threshold is not None:
            self.filter_settings.cooldown_threshold = cooldown_threshold

        self.filter_settings.minimum_priority = minimum_priority
        self.filter_settings.require_full_only = require_full_only

        self.logger.info(f"Updated filter settings: "
                        f"enabled_types={self.filter_settings.enabled_types}, "
                        f"minimum_priority={minimum_priority}, "
                        f"require_full_only={require_full_only}")

    def reset_filtering(self):
        """Reset all filtering settings to defaults"""
        self.filter_settings = ResourceTypeFilter()
        self.logger.info("Reset all filter settings to defaults")

    def add_resource_location(self, resource_type: str, position: Tuple[int, int], is_full: bool = True):
        """Add or update a resource node"""
        if position in self.nodes:
            node = self.nodes[position]
            node.is_full = is_full
            self.logger.debug(f"Updated {resource_type} at {position} (full: {is_full})")
        else:
            node = ResourceNode(position, resource_type, is_full)
            self.nodes[position] = node
            self.logger.debug(f"Added new {resource_type} at {position} (full: {is_full})")

    def set_node_priority(self, position: Tuple[int, int], priority: float):
        """Set priority for a specific resource node"""
        if position in self.nodes:
            self.nodes[position].priority = max(0.0, min(priority, 2.0))  # Clamp between 0-2

    def add_terrain_penalty(self, position: Tuple[int, int], penalty: float):
        """Mark area as difficult/dangerous to traverse"""
        if self.current_route:
            self.current_route.terrain_penalties[position] = max(0.0, min(penalty, 1.0))

    def create_route(self, 
                    resource_types: Optional[List[str]] = None,
                    start_pos: Optional[Tuple[int, int]] = None,
                    max_distance: Optional[float] = None,
                    complete_cycle: bool = True,
                    prefer_full: bool = True,
                    min_priority: float = 0.0) -> Optional[ResourceRoute]:
        """Create an optimized route through specified resource types with filtering"""
        try:
            # Validate start position
            if start_pos is None:
                self.logger.error("Start position is required")
                self.stats['failed_routes'] += 1
                return None

            route = ResourceRoute()
            route.cycle_complete = complete_cycle

            # Add start position
            start_node = ResourceNode(start_pos, "start")
            route.add_waypoint(start_node)

            # Apply resource type filtering if specified
            if resource_types:
                self.configure_filtering(enabled_types=resource_types,
                                      minimum_priority=min_priority,
                                      require_full_only=prefer_full)

            # Collect available nodes considering filters
            available_nodes = []
            filtered_count = 0
            for pos, node in self.nodes.items():
                if not node.is_available(self.filter_settings):
                    filtered_count += 1
                    continue
                if max_distance is not None:
                    distance = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
                    if distance > max_distance:
                        filtered_count += 1
                        continue
                available_nodes.append(node)

            self.stats['filtered_nodes'] = filtered_count

            if not available_nodes:
                self.logger.warning("No suitable resources found after filtering")
                self.stats['failed_routes'] += 1
                return None

            # Add available nodes to route
            for node in available_nodes:
                route.add_waypoint(node)

            # Optimize route
            route.optimize()

            # Calculate path and validate
            full_path = []
            bounds = self._calculate_bounds(route.waypoints)
            route_valid = True

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
                    self.logger.warning(f"Invalid path between {route.waypoints[i]} and {route.waypoints[i + 1]}")
                    route_valid = False
                    break

            if not route_valid:
                self.stats['failed_routes'] += 1
                return None

            # Calculate estimated time
            base_time = route.total_distance * 0.5  # Base movement time
            gathering_time = sum(2.0 + (node.visit_count * 0.5) for node in route.nodes)  # Increased time for revisits
            route.estimated_time = base_time + gathering_time

            self.current_route = route
            self.stats['total_routes'] += 1

            # Log detailed info
            self.logger.info(
                f"Created route with {len(route.waypoints)} waypoints "
                f"covering {len(route.resource_counts)} resource types\n"
                f"Distance: {route.total_distance:.1f} units\n"
                f"Est. time: {route.estimated_time:.1f}s\n"
                f"Resource distribution: {dict(route.resource_counts)}\n"
                f"Filtered nodes: {filtered_count}"
            )

            return route

        except Exception as e:
            self.logger.error(f"Error creating route: {str(e)}")
            self.stats['failed_routes'] += 1
            return None

    def record_completion(self, route: ResourceRoute, actual_time: float):
        """Record statistics for completed route"""
        self.stats['average_completion_time'] = (
            (self.stats['average_completion_time'] * (self.stats['total_routes'] - 1) + actual_time) / 
            self.stats['total_routes']
        )

        for node in route.nodes:
            if node.resource_type != 'start':
                self.stats['resources_collected'][node.resource_type] = (
                    self.stats['resources_collected'].get(node.resource_type, 0) + 1
                )
                node.visit()  # Update node state

    def visualize_route(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Create a visualization with filter indicators"""
        try:
            import cv2
            visualization = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

            if not self.current_route or not self.current_route.waypoints:
                # Draw message when no route is available
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(visualization, "No valid route available", 
                          (10, 30), font, 1, (255, 255, 255), 2)
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

            # Draw terrain penalties if any
            for pos, penalty in self.current_route.terrain_penalties.items():
                pos_scaled = (
                    int(pos[0] * image_size[0] / self.pathfinder.grid_size),
                    int(pos[1] * image_size[1] / self.pathfinder.grid_size)
                )
                color = (0, 0, int(255 * penalty))  # Red with alpha based on penalty
                cv2.circle(visualization, pos_scaled, 10, color, -1)

            # Draw connections between waypoints
            total_points = len(self.current_route.waypoints)
            if total_points > 1:  # Only draw connections if we have more than one point
                for i in range(total_points - 1):
                    start = self.current_route.waypoints[i]
                    end = self.current_route.waypoints[i + 1]

                    # Scale coordinates
                    start_scaled = (
                        int(start[0] * image_size[0] / self.pathfinder.grid_size),
                        int(start[1] * image_size[1] / self.pathfinder.grid_size)
                    )
                    end_scaled = (
                        int(end[0] * image_size[0] / self.pathfinder.grid_size),
                        int(end[1] * image_size[1] / self.pathfinder.grid_size)
                    )

                    # Draw path with gradient color based on sequence
                    # Use max to prevent division by zero
                    progress = i / max(1, total_points - 2)  # Prevent division by zero
                    path_color = (
                        int(255 * (1 - progress)),  # Blue decreases
                        int(255 * progress),        # Green increases
                        0
                    )
                    cv2.line(visualization, start_scaled, end_scaled, path_color, 2)

                    # Draw resource markers
                    node = self.current_route.nodes[i]
                    base_color = type_colors.get(node.resource_type, (0, 255, 255))

                    # Modify color based on priority
                    priority_factor = node.get_effective_priority(self.filter_settings)
                    color = tuple(int(min(c * priority_factor, 255)) for c in base_color)  # Clamp to 255

                    # Draw markers
                    if node.resource_type == 'start':
                        cv2.circle(visualization, start_scaled, 8, color, -1)
                        # Draw start label
                        cv2.putText(visualization, "S", 
                                  (start_scaled[0] - 5, start_scaled[1] + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Draw node with visit count indicator
                        if node.is_available(self.filter_settings):
                            cv2.circle(visualization, start_scaled, 6, color, -1)
                        else:
                            cv2.circle(visualization, start_scaled, 6, color, 2)

                        if node.visit_count > 0:
                            cv2.putText(visualization, str(node.visit_count),
                                      (start_scaled[0] - 5, start_scaled[1] - 8),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw final point if it exists
            if self.current_route.nodes:
                final_node = self.current_route.nodes[-1]
                final_color = type_colors.get(final_node.resource_type, (0, 255, 255))
                final_pos = final_node.position
                final_scaled = (
                    int(final_pos[0] * image_size[0] / self.pathfinder.grid_size),
                    int(final_pos[1] * image_size[1] / self.pathfinder.grid_size)
                )
                cv2.circle(visualization, final_scaled, 6, final_color, -1)

            # Add legend
            legend_y = 20
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Route Statistics
            cv2.putText(visualization, "Route Statistics:", (10, legend_y),
                      font, 0.5, (255, 255, 255), 1)
            legend_y += 20
            cv2.putText(visualization, f"Total routes: {self.stats['total_routes']}", 
                      (10, legend_y), font, 0.5, (255, 255, 255), 1)
            legend_y += 20
            cv2.putText(visualization, f"Avg time: {self.stats['average_completion_time']:.1f}s",
                      (10, legend_y), font, 0.5, (255, 255, 255), 1)
            legend_y += 20

            # Filter Status
            cv2.putText(visualization, "Filter Status:", (10, legend_y),
                      font, 0.5, (255, 255, 255), 1)
            legend_y += 20

            if self.filter_settings.enabled_types:
                types_str = ", ".join(sorted(self.filter_settings.enabled_types))
                cv2.putText(visualization, f"Enabled: {types_str}", 
                          (10, legend_y), font, 0.4, (0, 255, 0), 1)
                legend_y += 15

            if self.filter_settings.require_full_only:
                cv2.putText(visualization, "Full nodes only",
                          (10, legend_y), font, 0.4, (0, 255, 255), 1)
                legend_y += 15

            if self.stats['filtered_nodes'] > 0:
                cv2.putText(visualization, f"Filtered: {self.stats['filtered_nodes']} nodes",
                          (10, legend_y), font, 0.4, (255, 165, 0), 1)

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