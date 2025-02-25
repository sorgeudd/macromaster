"""Test script for resource route management"""
import logging
import cv2
import numpy as np
from vision_system import VisionSystem
from resource_route_manager import ResourceRouteManager
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('RouteTest')

def test_resource_routing():
    """Test resource route creation and visualization"""
    try:
        # Initialize systems
        vision = VisionSystem()
        route_manager = ResourceRouteManager()

        # Test with sample image
        img_path = 'attached_assets/image_1740444431862.png'
        if not Path(img_path).exists():
            logger.error(f"Test image not found: {img_path}")
            return False

        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            logger.error("Failed to load test image")
            return False

        # Detect resources
        detections = vision.detect_resources(img)
        logger.info(f"Detected {len(detections)} resources")

        # Add detected resources to route manager
        for det in detections:
            if det['bbox']:
                # Convert bbox center to grid position
                x, y, w, h = map(int, det['bbox'])
                grid_x = x // route_manager.pathfinder.grid_size
                grid_y = y // route_manager.pathfinder.grid_size
                route_manager.add_resource_location(det['class_id'], (grid_x, grid_y))

        # Create route starting from center of image
        start_pos = (img.shape[1] // (2 * route_manager.pathfinder.grid_size),
                    img.shape[0] // (2 * route_manager.pathfinder.grid_size))

        route = route_manager.create_route(
            resource_types=['titanium', 'ore', 'stone'],
            start_pos=start_pos
        )

        if route:
            # Create visualization
            route_viz = route_manager.visualize_route((img.shape[1], img.shape[0]))

            # Overlay route on original image
            overlay = cv2.addWeighted(img, 0.7, route_viz, 0.3, 0)

            # Add route information
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, f"Route length: {route.total_distance:.1f} units", 
                      (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(overlay, f"Estimated time: {route.estimated_time:.1f}s",
                      (10, 60), font, 1, (0, 255, 0), 2)

            # Save result
            output_path = 'resource_route_visualization.png'
            cv2.imwrite(output_path, overlay)
            logger.info(f"Saved route visualization to: {output_path}")

            return True
        else:
            logger.error("Failed to create route")
            return False

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