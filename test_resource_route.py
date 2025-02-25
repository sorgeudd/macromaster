"""Test script for resource route management"""
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
    """Test resource route creation and visualization"""
    try:
        # Initialize systems
        vision = VisionSystem()
        route_manager = ResourceRouteManager()

        # Test with game screenshot containing both copper and limestone
        img_path = 'attached_assets/image_1740461879286.png'
        if not Path(img_path).exists():
            logger.error(f"Test image not found: {img_path}")
            return False

        # Load test image
        img = cv2.imread(img_path)
        if img is None:
            logger.error("Failed to load test image")
            return False

        # Add example resource locations with varied priorities
        # Copper nodes
        route_manager.add_resource_location('copper_ore_full', (20, 30), True)
        route_manager.set_node_priority((20, 30), 1.5)  # High priority

        route_manager.add_resource_location('copper_ore_empty', (35, 45), False)
        route_manager.set_node_priority((35, 45), 0.5)  # Low priority

        route_manager.add_resource_location('copper_ore_full', (50, 35), True)
        route_manager.set_node_priority((50, 35), 1.2)  # Medium-high priority

        # Limestone nodes
        route_manager.add_resource_location('limestone_full', (60, 25), True)
        route_manager.set_node_priority((60, 25), 1.8)  # Highest priority

        route_manager.add_resource_location('limestone_empty', (75, 40), False)
        route_manager.set_node_priority((75, 40), 0.3)  # Lowest priority

        route_manager.add_resource_location('limestone_full', (45, 55), True)
        route_manager.set_node_priority((45, 55), 1.0)  # Normal priority

        # Add some terrain penalties
        route_manager.add_terrain_penalty((40, 40), 0.8)  # High risk area
        route_manager.add_terrain_penalty((55, 30), 0.5)  # Medium risk area

        # Create route starting from center of image
        start_pos = (img.shape[1] // (2 * route_manager.pathfinder.grid_size),
                    img.shape[0] // (2 * route_manager.pathfinder.grid_size))

        # Test different route configurations
        test_cases = [
            {
                'name': 'Full resources only',
                'types': ['copper_ore_full', 'limestone_full'],
                'prefer_full': True,
                'min_priority': 1.0  # Only high priority nodes
            },
            {
                'name': 'All copper nodes',
                'types': ['copper_ore_full', 'copper_ore_empty'],
                'prefer_full': False,
                'min_priority': 0.0  # Any priority
            },
            {
                'name': 'Mixed resources',
                'types': ['copper_ore_full', 'copper_ore_empty', 'limestone_full', 'limestone_empty'],
                'prefer_full': False,
                'min_priority': 0.5  # Medium priority and up
            }
        ]

        for test_case in test_cases:
            logger.info(f"\nTesting route: {test_case['name']}")
            route = route_manager.create_route(
                resource_types=test_case['types'],
                start_pos=start_pos,
                prefer_full=test_case['prefer_full'],
                min_priority=test_case['min_priority']
            )

            if route:
                # Simulate route completion for statistics
                start_time = time()
                sleep(0.1)  # Simulate some processing time
                route_manager.record_completion(route, time() - start_time)

                # Create visualization
                route_viz = route_manager.visualize_route((img.shape[1], img.shape[0]))

                # Overlay route on original image
                overlay = cv2.addWeighted(img, 0.7, route_viz, 0.3, 0)

                # Add route information
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_pos = 30
                cv2.putText(overlay, f"Route: {test_case['name']}", 
                          (10, y_pos), font, 1, (255, 255, 255), 2)
                y_pos += 30
                cv2.putText(overlay, f"Distance: {route.total_distance:.1f} units", 
                          (10, y_pos), font, 1, (0, 255, 0), 2)
                y_pos += 30
                cv2.putText(overlay, f"Est. time: {route.estimated_time:.1f}s",
                          (10, y_pos), font, 1, (0, 255, 0), 2)
                y_pos += 30
                cv2.putText(overlay, f"Min priority: {test_case['min_priority']:.1f}",
                          (10, y_pos), font, 1, (0, 255, 255), 2)

                # Save result
                output_path = f"resource_route_{test_case['name'].lower().replace(' ', '_')}.png"
                cv2.imwrite(output_path, overlay)
                logger.info(f"Saved route visualization to: {output_path}")

                # Log route details
                logger.info("Route Details:")
                for res_type, count in route.resource_counts.items():
                    logger.info(f"  {res_type}: {count} nodes")
            else:
                logger.error(f"Failed to create route for: {test_case['name']}")
                return False

        # Log overall statistics
        logger.info("\nOverall Statistics:")
        logger.info(f"Total routes created: {route_manager.stats['total_routes']}")
        logger.info(f"Average completion time: {route_manager.stats['average_completion_time']:.2f}s")
        logger.info(f"Failed routes: {route_manager.stats['failed_routes']}")
        logger.info("Resources collected:")
        for res_type, count in route_manager.stats['resources_collected'].items():
            logger.info(f"  {res_type}: {count}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_resource_routing()
    if success:
        logger.info("Resource routing test completed successfully")
    else:
        logger.error("Resource routing test failed")
        exit(1)