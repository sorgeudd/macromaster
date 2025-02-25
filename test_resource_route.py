"""Test script for resource route management with terrain awareness"""
import logging
import cv2
import numpy as np
from vision_system import VisionSystem
from resource_route_manager import ResourceRouteManager
from pathlib import Path
from time import time, sleep

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('RouteTest')

def test_resource_routing():
    """Test terrain-aware resource route creation and visualization"""
    try:
        # Initialize systems
        vision = VisionSystem()
        route_manager = ResourceRouteManager()

        # Load terrain map from maps directory
        map_path = Path("maps/default_map.png")
        if not map_path.exists():
            logger.error(f"Default map not found. Falling back to asset directory")
            map_path = Path("attached_assets/image_1740466625160.png")
            if not map_path.exists():
                logger.error("No map file found")
                return False

        # Load and process map
        img = cv2.imread(str(map_path))
        if img is None:
            logger.error("Failed to load map image")
            return False

        # Initialize map with terrain data
        logger.info("Loading map for terrain-aware routing")
        success = route_manager.load_area_map("test_map", img)
        if not success:
            logger.error("Failed to load area map")
            return False

        # Add resource nodes avoiding water areas
        logger.info("Adding resource locations avoiding water")
        test_positions = [
            ((20, 30), 1.5),  # High priority
            ((50, 35), 1.2),  # Medium-high priority
            ((45, 55), 1.0),  # Normal priority
        ]

        for pos, priority in test_positions:
            # Check terrain before adding node
            terrain = route_manager.map_manager.detect_terrain(pos)
            if terrain['water'] < 0.3:  # Only add if not in water
                route_manager.add_resource_location('copper_ore_full', pos, True)
                route_manager.set_node_priority(pos, priority)
                logger.info(f"Added resource at {pos} with priority {priority}")
            else:
                logger.warning(f"Skipped position {pos} - water detected")

        # Calculate start position from center of map
        start_pos = (img.shape[1] // (2 * route_manager.pathfinder.grid_size),
                    img.shape[0] // (2 * route_manager.pathfinder.grid_size))

        # Test terrain-aware optimal route
        logger.info("\nTesting terrain-aware optimal route")
        logger.info("Resource types: ['copper_ore_full']")
        logger.info("Priority modifiers: {'copper_ore_full': 1.2}")
        logger.info("Min priority: 1.0")

        # Configure route settings
        route_manager.configure_filtering(
            enabled_types=['copper_ore_full'],
            priority_modifiers={'copper_ore_full': 1.2},
            minimum_priority=1.0,
            require_full_only=True
        )

        # Create route with terrain awareness
        route = route_manager.create_route(
            start_pos=start_pos,
            max_distance=None,
            complete_cycle=True,
            pattern="optimal",
            avoid_water=True
        )

        if route:
            # Record completion time
            start_time = time()
            sleep(0.1)  # Simulate processing time
            route_manager.record_completion(route, time() - start_time)

            # Create visualization with terrain overlay
            route_viz = route_manager.visualize_route((img.shape[1], img.shape[0]))

            # Blend with original map for better terrain visualization
            overlay = cv2.addWeighted(img, 0.7, route_viz, 0.3, 0)

            # Add route information
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_pos = 30
            cv2.putText(overlay, "Terrain-Aware Resource Route", 
                      (10, y_pos), font, 1, (255, 255, 255), 2)
            y_pos += 30
            cv2.putText(overlay, f"Distance: {route.total_distance:.1f} units", 
                      (10, y_pos), font, 1, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(overlay, f"Est. time: {route.estimated_time:.1f}s",
                      (10, y_pos), font, 1, (0, 255, 0), 2)

            # Save visualization
            output_path = "resource_route_terrain_aware.png"
            cv2.imwrite(output_path, overlay)
            logger.info(f"Saved route visualization to: {output_path}")

            # Log route details
            logger.info("\nRoute Details:")
            logger.info(f"Total distance: {route.total_distance:.1f} units")
            logger.info(f"Estimated time: {route.estimated_time:.1f}s")
            logger.info("Resource distribution:")
            for res_type, count in route.resource_counts.items():
                logger.info(f"  {res_type}: {count} nodes")
            logger.info(f"Terrain penalties encountered: {len(route.terrain_penalties)}")
            logger.info(f"Waypoints: {len(route.waypoints)}")

            return True
        else:
            logger.error("Failed to create terrain-aware route")
            logger.error("Route generation failed - check terrain penalties")
            return False

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_resource_routing()
    if success:
        logger.info("Resource routing test completed successfully")
    else:
        logger.error("Resource routing test failed")
        exit(1)