"""Resource route management system for automated farming"""
import logging
from typing import List, Tuple, Dict, Optional, Set, Literal
from pathfinding import PathFinder
import numpy as np
from time import time
from map_manager import MapManager

class ResourceTypeFilter:
    def __init__(self):
        self.enabled_types: Set[str] = set()
        self.priority_modifiers: Dict[str, float] = {}
        self.minimum_priority: float = 0.0
        self.require_full_only: bool = False
        self.cooldown_threshold: float = 300.0

    def is_type_enabled(self, resource_type: str) -> bool:
        return not self.enabled_types or resource_type in self.enabled_types

    def get_priority_modifier(self, resource_type: str) -> float:
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
        self.filtered_out = False
        self.respawn_time = 300.0  # Default 5 minute respawn
        self.last_seen_full = time() if is_full else 0.0

    def is_available(self, filter_settings: Optional[ResourceTypeFilter] = None) -> bool:
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
        if not filter_settings:
            return self.priority
        base_priority = self.priority * filter_settings.get_priority_modifier(self.resource_type)

        # Adjust priority based on respawn timing
        time_since_visit = time() - self.last_visited
        if time_since_visit >= self.respawn_time:
            return base_priority * 1.2  # 20% bonus for fully respawned nodes
        return base_priority

    def update_state(self, is_full: bool):
        """Update node state and track respawn timing"""
        if is_full != self.is_full:
            if is_full:
                # Node has respawned
                actual_respawn_time = time() - self.last_seen_full
                # Update respawn time estimate (moving average)
                self.respawn_time = (self.respawn_time * 0.8) + (actual_respawn_time * 0.2)
            else:
                self.last_seen_full = time()
        self.is_full = is_full

    def visit(self):
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
        self.nodes: List[ResourceNode] = []
        self.terrain_penalties: Dict[Tuple[int, int], float] = {}
        self.pattern: str = "optimal"  # optimal, circular, zigzag
        self.density_map: Dict[Tuple[int, int], float] = {}  # Resource density heatmap

    def add_waypoint(self, node: ResourceNode):
        self.waypoints.append(node.position)
        self.resource_types.append(node.resource_type)
        self.nodes.append(node)
        self.resource_counts[node.resource_type] = self.resource_counts.get(node.resource_type, 0) + 1

    def update_density_map(self, radius: int = 3):
        """Update resource density heatmap"""
        self.density_map.clear()
        for node in self.nodes:
            pos = node.position
            # Add density value to surrounding area
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist = abs(dx) + abs(dy)
                    if dist <= radius:
                        density_pos = (pos[0] + dx, pos[1] + dy)
                        weight = 1.0 - (dist / (radius + 1))
                        self.density_map[density_pos] = self.density_map.get(density_pos, 0.0) + weight

    def get_density_score(self, position: Tuple[int, int]) -> float:
        """Get resource density score for a position"""
        return self.density_map.get(position, 0.0)

    def optimize(self, pattern: str = "optimal", 
                distance_weight: float = 0.6,
                variety_weight: float = 0.2, 
                priority_weight: float = 0.2,
                density_weight: float = 0.1):
        """Optimize route using specified pattern and weights"""
        if len(self.waypoints) <= 2:
            return

        self.pattern = pattern
        self.update_density_map()

        # Start with first point
        optimized_waypoints = [self.waypoints[0]]
        optimized_types = [self.resource_types[0]]
        optimized_nodes = [self.nodes[0]]
        remaining_indices = list(range(1, len(self.waypoints)))
        current = self.waypoints[0]
        visited_types = {self.resource_types[0]: 1}

        # Adjust weights based on pattern
        pattern_weight = 0.3  # Increase pattern influence
        if pattern == "circular":
            distance_weight *= 0.8  # Reduce distance importance for circular routes
            pattern_weight = 0.4
        elif pattern == "zigzag":
            variety_weight *= 0.8  # Reduce variety importance for zigzag
            pattern_weight = 0.4

        while remaining_indices:
            scores = []
            for i in remaining_indices:
                node = self.nodes[i]
                pos = node.position

                # Distance score (lower is better)
                distance = abs(pos[0] - current[0]) + abs(pos[1] - current[1])
                distance_score = 1.0 - min(distance / 100.0, 1.0)

                # Variety score (higher is better)
                type_count = visited_types.get(node.resource_type, 0)
                variety_score = 1.0 / (type_count + 1)

                # Priority score
                priority_score = node.get_effective_priority(None)

                # Density score
                density_score = self.get_density_score(pos)

                # Pattern-specific scoring
                pattern_score = self.get_pattern_score(pos, current, pattern)

                # Terrain penalty
                terrain_penalty = self.terrain_penalties.get(pos, 0.0)

                # Combined score (higher is better)
                total_score = (
                    distance_score * distance_weight +
                    variety_score * variety_weight +
                    priority_score * priority_weight +
                    density_score * density_weight +
                    pattern_score * pattern_weight
                ) * (1.0 - terrain_penalty)

                scores.append((total_score, i))

            best_score, best_idx = max(scores)
            chosen_idx = remaining_indices[best_idx]
            chosen_node = self.nodes[chosen_idx]

            optimized_waypoints.append(chosen_node.position)
            optimized_types.append(chosen_node.resource_type)
            optimized_nodes.append(chosen_node)
            visited_types[chosen_node.resource_type] = visited_types.get(chosen_node.resource_type, 0) + 1
            current = chosen_node.position
            remaining_indices.remove(chosen_idx)

        if self.cycle_complete and optimized_waypoints:
            optimized_waypoints.append(optimized_waypoints[0])
            optimized_types.append(optimized_types[0])
            optimized_nodes.append(optimized_nodes[0])

        self.waypoints = optimized_waypoints
        self.resource_types = optimized_types
        self.nodes = optimized_nodes

    def get_pattern_score(self, pos: Tuple[int, int], current: Tuple[int, int], pattern: str) -> float:
        """Calculate score based on how well position fits desired pattern"""
        if pattern == "optimal":
            return 1.0  # No pattern adjustment
        elif pattern == "circular":
            # Favor positions that maintain similar distance from start
            if not self.waypoints:
                return 1.0
            start_pos = self.waypoints[0]
            start_dist = abs(start_pos[0] - current[0]) + abs(start_pos[1] - current[1])
            pos_dist = abs(start_pos[0] - pos[0]) + abs(start_pos[1] - pos[1])
            return 1.0 - min(abs(start_dist - pos_dist) / 50.0, 1.0)
        elif pattern == "zigzag":
            # Favor alternating left-right movement
            if len(self.waypoints) < 2:
                return 1.0
            dx = pos[0] - current[0]
            prev_pos = self.waypoints[-1]
            last_dx = current[0] - prev_pos[0]
            # Factor in y-movement to encourage forward progress
            dy = pos[1] - current[1]
            prev_dy = current[1] - prev_pos[1]

            # Score for horizontal zigzag
            direction_change = 1.0 if (dx * last_dx) <= 0 else 0.5

            # Score for vertical progress
            vertical_progress = 0.8 if dy > 0 else (0.6 if dy == 0 else 0.4)

            # Penalize backtracking
            backtrack_penalty = 0.7 if dy < 0 and prev_dy < 0 else 1.0

            return direction_change * vertical_progress * backtrack_penalty
        return 1.0

class ResourceRouteManager:
    def __init__(self, grid_size: int = 32):
        self.logger = logging.getLogger('ResourceRouteManager')
        self.pathfinder = PathFinder(grid_size)
        self.map_manager = MapManager()
        self.current_route: Optional[ResourceRoute] = None
        self.nodes: Dict[Tuple[int, int], ResourceNode] = {}
        self.stats: Dict[str, Dict] = {
            'total_routes': 0,
            'resources_collected': {},
            'average_completion_time': 0.0,
            'failed_routes': 0,
            'filtered_nodes': 0,
            'terrain_blocked_paths': 0
        }
        self.filter_settings = ResourceTypeFilter()

    def load_area_map(self, map_name: Optional[str] = None, minimap_image: Optional[np.ndarray] = None) -> bool:
        """Load appropriate map for current area"""
        try:
            if minimap_image is not None:
                detected_map = self.map_manager.extract_map_name_from_minimap(minimap_image)
                if detected_map:
                    return self.map_manager.load_map(detected_map)

            if map_name:
                return self.map_manager.load_map(map_name)

            return False

        except Exception as e:
            self.logger.error(f"Error loading area map: {str(e)}")
            return False

    def add_terrain_penalty(self, position: Tuple[int, int], penalty: float):
        """Add manual terrain penalty or use map-based detection"""
        if self.current_route:
            map_penalty = self.map_manager.get_terrain_penalty(position)
            combined_penalty = max(penalty, map_penalty)
            self.current_route.terrain_penalties[position] = max(0.0, min(combined_penalty, 1.0))

    def create_route(self,
                    resource_types: Optional[List[str]] = None,
                    start_pos: Optional[Tuple[int, int]] = None,
                    max_distance: Optional[float] = None,
                    complete_cycle: bool = True,
                    prefer_full: bool = True,
                    min_priority: float = 0.0,
                    pattern: str = "optimal",
                    avoid_water: bool = True) -> Optional[ResourceRoute]:
        """Create terrain-aware resource route"""
        try:
            if start_pos is None:
                self.logger.error("Start position is required")
                self.stats['failed_routes'] += 1
                return None

            # Check if starting position is in water
            if avoid_water:
                terrain = self.map_manager.detect_terrain(start_pos)
                if terrain['water'] > 0.5:
                    self.logger.warning("Start position is in water")
                    self.stats['terrain_blocked_paths'] += 1
                    return None

            route = ResourceRoute()
            route.cycle_complete = complete_cycle
            route.pattern = "optimal"  # Force optimal pattern as selected

            start_node = ResourceNode(start_pos, "start")
            route.add_waypoint(start_node)

            if resource_types:
                self.configure_filtering(enabled_types=resource_types,
                                      minimum_priority=min_priority,
                                      require_full_only=prefer_full)

            available_nodes = []
            filtered_count = 0
            for pos, node in self.nodes.items():
                # Check terrain penalties
                terrain = self.map_manager.detect_terrain(pos)
                terrain_penalty = self.map_manager.get_terrain_penalty(pos)

                # Skip nodes in impassable terrain
                if terrain['water'] > 0.5 or terrain_penalty > 0.9:
                    filtered_count += 1
                    continue

                if not node.is_available(self.filter_settings):
                    filtered_count += 1
                    continue

                if max_distance is not None:
                    distance = abs(pos[0] - start_pos[0]) + abs(pos[1] - start_pos[1])
                    if distance > max_distance:
                        filtered_count += 1
                        continue

                # Add terrain penalty to node
                self.add_terrain_penalty(pos, terrain_penalty)
                available_nodes.append(node)

            self.stats['filtered_nodes'] = filtered_count

            if not available_nodes:
                self.logger.warning("No suitable resources found after filtering")
                self.stats['failed_routes'] += 1
                return None

            for node in available_nodes:
                route.add_waypoint(node)

            # Enhanced optimization weights for terrain awareness
            route.optimize(
                pattern="optimal",
                distance_weight=0.5,  # Reduced to give more weight to terrain
                variety_weight=0.2,
                priority_weight=0.2,
                density_weight=0.1
            )

            full_path = []
            bounds = self._calculate_bounds(route.waypoints)
            route_valid = True

            for i in range(len(route.waypoints) - 1):
                # Find path considering terrain
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

            # Adjust time estimates based on terrain
            base_time = route.total_distance * 0.5
            terrain_modifier = 1.0 + sum(penalty for penalty in route.terrain_penalties.values()) / len(route.terrain_penalties) if route.terrain_penalties else 1.0
            gathering_time = sum(2.0 + (node.visit_count * 0.5) for node in route.nodes)
            route.estimated_time = (base_time + gathering_time) * terrain_modifier

            self.current_route = route
            self.stats['total_routes'] += 1

            self.logger.info(
                f"Created optimal route with {len(route.waypoints)} waypoints\n"
                f"Distance: {route.total_distance:.1f} units\n"
                f"Est. time: {route.estimated_time:.1f}s\n"
                f"Terrain modifier: {terrain_modifier:.2f}\n"
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
        """Enhanced visualization with terrain overlay"""
        try:
            import cv2
            visualization = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

            # Add terrain overlay if map is loaded
            if self.map_manager.current_map is not None:
                # Scale map to match visualization size
                terrain_overlay = cv2.resize(
                    self.map_manager.current_map,
                    (image_size[0], image_size[1])
                )
                # Blend with 30% opacity
                visualization = cv2.addWeighted(
                    visualization, 0.7,
                    terrain_overlay, 0.3,
                    0
                )

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

            # Draw pattern-specific visualization
            if self.current_route.pattern == "circular":
                # Draw circular guide
                if len(self.current_route.waypoints) > 1:
                    center = self.current_route.waypoints[0]
                    center_scaled = (
                        int(center[0] * image_size[0] / self.pathfinder.grid_size),
                        int(center[1] * image_size[1] / self.pathfinder.grid_size)
                    )
                    avg_radius = sum(
                        abs(pos[0] - center[0]) + abs(pos[1] - center[1])
                        for pos in self.current_route.waypoints[1:]
                    ) / (len(self.current_route.waypoints) - 1)
                    radius_scaled = int(avg_radius * image_size[0] / self.pathfinder.grid_size)
                    cv2.circle(visualization, center_scaled, radius_scaled, (50, 50, 50), 1)

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
                    progress = i / max(1, total_points - 2)  # Prevent division by zero
                    path_color = (
                        int(255 * (1 - progress)),  # Blue decreases
                        int(255 * progress),        # Green increases
                        0
                    )
                    cv2.line(visualization, start_scaled, end_scaled, path_color, 2)

                    # Draw direction arrows for zigzag pattern
                    if self.current_route.pattern == "zigzag":
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        mid_x = (start_scaled[0] + end_scaled[0]) // 2
                        mid_y = (start_scaled[1] + end_scaled[1]) // 2
                        arrow_color = (200, 200, 0)  # Yellow arrows
                        # Draw arrowhead
                        cv2.arrowedLine(visualization, 
                                      (mid_x - 5, mid_y), 
                                      (mid_x + 5, mid_y), 
                                      arrow_color, 1, tipLength=0.5)

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
            node.update_state(is_full)
            self.logger.debug(f"Updated {resource_type} at {position} (full: {is_full})")
        else:
            node = ResourceNode(position, resource_type, is_full)
            self.nodes[position] = node
            self.logger.debug(f"Added new {resource_type} at {position} (full: {is_full})")

    def set_node_priority(self, position: Tuple[int, int], priority: float):
        """Set priority for a specific resource node"""
        if position in self.nodes:
            self.nodes[position].priority = max(0.0, min(priority, 2.0))  # Clamp between 0-2

    def _calculate_bounds(self, positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate map bounds based on positions"""
        if not positions:
            return (100, 100)  # Default size

        x_coords, y_coords = zip(*positions)
        max_x = max(x_coords) + 50  # Add margin
        max_y = max(y_coords) + 50
        return (max_x, max_y)