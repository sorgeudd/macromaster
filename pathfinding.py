"""A* pathfinding system for game navigation"""
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import heapq

@dataclass
class Node:
    position: Tuple[int, int]
    g_cost: float = float('inf')  # Cost from start to this node
    h_cost: float = float('inf')  # Estimated cost to goal
    parent: Optional['Node'] = None

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class PathFinder:
    def __init__(self, grid_size=32):
        self.logger = logging.getLogger('PathFinder')
        self.grid_size = grid_size
        self.obstacles = set()
        self.map_data = None
        self.directions = [
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1),  # Left
            (-1, 0),  # Up
            (1, 1),   # Diagonal
            (-1, 1),
            (1, -1),
            (-1, -1)
        ]

    def update_map(self, map_data: Dict[str, Any]) -> None:
        """Update map data and process walkable areas
        Args:
            map_data: Dictionary containing map information
                For PNG maps: {'binary_map': List[List[bool]], 'resolution': int}
                For JSON/CSV: {'nodes': List[Dict], 'edges': List[Dict]}
        """
        try:
            self.map_data = map_data
            if 'binary_map' in map_data:
                # Convert list to numpy array if needed
                binary_map = map_data['binary_map']
                if isinstance(binary_map, list):
                    binary_map = np.array(binary_map)

                resolution = map_data.get('resolution', self.grid_size)

                # Update obstacles based on binary map
                height, width = binary_map.shape
                self.obstacles.clear()
                for y in range(height):
                    for x in range(width):
                        if not binary_map[y, x]:  # Non-walkable area
                            self.obstacles.add((x, y))

                self.logger.info(f"Updated map from PNG: {len(self.obstacles)} obstacles")

            else:
                # Process JSON/CSV map data
                nodes = map_data.get('nodes', [])
                self.obstacles.clear()

                # Extract obstacles from node data
                for node in nodes:
                    if node.get('type') == 'obstacle':
                        self.obstacles.add((node['x'], node['y']))

                self.logger.info(f"Updated map from nodes: {len(self.obstacles)} obstacles")

        except Exception as e:
            self.logger.error(f"Error updating map: {str(e)}")
            raise

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int], bounds: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        max_x, max_y = bounds

        for dx, dy in self.directions:
            new_x = pos[0] + dx
            new_y = pos[1] + dy

            # Check bounds
            if 0 <= new_x < max_x and 0 <= new_y < max_y:
                new_pos = (new_x, new_y)
                # Check if position is obstacle-free
                if new_pos not in self.obstacles:
                    neighbors.append(new_pos)

        return neighbors

    def update_obstacles(self, obstacles: List[Tuple[int, int]]):
        """Update obstacle positions"""
        self.obstacles = set(obstacles)
        self.logger.info(f"Updated obstacles: {len(self.obstacles)} positions")

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                 bounds: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path using A* algorithm"""
        if start == goal:
            return [start]

        # Initialize nodes
        start_node = Node(start, g_cost=0, h_cost=self._heuristic(start, goal))
        open_set = []
        closed_set = set()

        # Add start node
        heapq.heappush(open_set, start_node)
        nodes = {start: start_node}

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal:
                # Reconstruct path
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            closed_set.add(current.position)

            # Check neighbors
            for neighbor_pos in self._get_neighbors(current.position, bounds):
                if neighbor_pos in closed_set:
                    continue

                # Calculate new cost
                new_g_cost = current.g_cost + 1

                if neighbor_pos not in nodes:
                    neighbor = Node(
                        neighbor_pos,
                        g_cost=new_g_cost,
                        h_cost=self._heuristic(neighbor_pos, goal),
                        parent=current
                    )
                    nodes[neighbor_pos] = neighbor
                    heapq.heappush(open_set, neighbor)
                else:
                    neighbor = nodes[neighbor_pos]
                    if new_g_cost < neighbor.g_cost:
                        neighbor.g_cost = new_g_cost
                        neighbor.parent = current

        self.logger.warning("No path found")
        return []

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Smooth path to remove unnecessary zigzags"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Look ahead for furthest visible point
            look_ahead = current_idx + 2
            while look_ahead < len(path):
                # Check if direct path is clear
                if self._is_path_clear(path[current_idx], path[look_ahead]):
                    look_ahead += 1
                else:
                    look_ahead -= 1
                    break

            smoothed.append(path[look_ahead])
            current_idx = look_ahead

        return smoothed

    def _is_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if direct path between points is clear of obstacles"""
        x0, y0 = start
        x1, y1 = end

        # Use Bresenham's line algorithm to check path
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if (x, y) in self.obstacles:
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True