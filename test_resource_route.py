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
        success = route_manager.map_manager.load_map("test_map", img)
        if not success:
            logger.error("Failed to load area map")
            return False

        # Test positions in different terrain types
        test_positions = [
            ((20, 30), 1.5, "land"),      # High priority on land
            ((50, 35), 1.2, "shallow"),   # Medium priority in shallow water
            ((45, 55), 1.0, "land"),      # Normal priority on land
            ((60, 40), 0.8, "deep"),      # Low priority near deep water
        ]

        logger.info("Testing terrain detection and resource placement:")
        for pos, priority, expected_terrain in test_positions:
            # Check terrain before adding node
            terrain = route_manager.map_manager.detect_terrain(pos)

            # Log terrain detection results
            logger.info(f"\nPosition {pos} (expected {expected_terrain}):")
            logger.info(f"Deep water: {terrain['deep_water']:.2f}")
            logger.info(f"Shallow water: {terrain['shallow_water']:.2f}")
            logger.info(f"Mountain: {terrain['mountain']:.2f}")
            logger.info(f"Cliff: {terrain['cliff']:.2f}")

            # Skip deep water, allow shallow water
            if terrain['deep_water'] > 0.3:
                logger.warning(f"Skipped position {pos} - deep water detected")
                continue

            # Add resource with appropriate priority
            route_manager.add_resource_location('copper_ore_full', pos, True)
            route_manager.set_node_priority(pos, priority)

            # Adjust priority based on terrain
            if terrain['shallow_water'] > 0.3:
                # Reduce priority slightly in shallow water
                adjusted_priority = priority * 0.8
                route_manager.set_node_priority(pos, adjusted_priority)
                logger.info(f"Added resource at {pos} with adjusted priority {adjusted_priority:.2f} (shallow water)")
            else:
                logger.info(f"Added resource at {pos} with priority {priority:.2f}")

        # Calculate start position from center of map
        start_pos = (img.shape[1] // (2 * route_manager.pathfinder.grid_size),
                    img.shape[0] // (2 * route_manager.pathfinder.grid_size))

        # Configure route settings
        logger.info("\nCreating terrain-aware route:")
        route_manager.configure_filtering(
            enabled_types=['copper_ore_full'],
            priority_modifiers={'copper_ore_full': 1.2},
            minimum_priority=0.8,  # Lower threshold to include shallow water resources
            require_full_only=True
        )

        # Create route with terrain awareness
        route = route_manager.create_route(
            start_pos=start_pos,
            max_distance=None,
            complete_cycle=True,
            pattern="optimal",
            avoid_water=True  # This now only avoids deep water
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
            logger.info(f"Nodes in shallow water: {sum(1 for pos in route.waypoints if route_manager.map_manager.detect_terrain(pos)['shallow_water'] > 0.3)}")
            logger.info(f"Total waypoints: {len(route.waypoints)}")

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