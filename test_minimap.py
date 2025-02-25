"""Test script for minimap player position detection and resource tracking"""
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from map_manager import MapManager, ResourceNode

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MinimapTest')

def create_test_map_data():
    """Create test map and resource data"""
    # Create test map directory
    test_map_dir = Path('maps')
    test_map_dir.mkdir(exist_ok=True)

    # Create test map image (200x200 with different terrain colors)
    map_size = (200, 200, 3)
    test_map = np.zeros(map_size, dtype=np.uint8)

    # Add terrain features
    cv2.rectangle(test_map, (0, 0), (100, 100), (50, 150, 50), -1)  # Forest
    cv2.rectangle(test_map, (100, 0), (200, 100), (150, 50, 50), -1)  # Mountain
    cv2.rectangle(test_map, (0, 100), (100, 200), (150, 150, 50), -1)  # Normal
    cv2.rectangle(test_map, (100, 100), (200, 200), (150, 150, 150), -1)  # Water

    # Save test map
    cv2.imwrite('maps/test_map.png', test_map)

    # Create test resource data
    resource_data = {
        "nodes": [
            {"x": 50, "y": 50, "type": "herb", "priority": 1.0},
            {"x": 150, "y": 150, "type": "fish", "priority": 0.8},
            {"x": 100, "y": 100, "type": "ore", "priority": 1.2}
        ]
    }

    # Save resource data
    with open('maps/test_map_resources.json', 'w') as f:
        json.dump(resource_data, f, indent=2)

    return test_map

def create_test_minimap(arrow_pos=(100, 100), arrow_angle=0):
    """Create a test minimap with player arrow"""
    # Create black background
    minimap_size = (200, 200, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Create darker background for better contrast
    cv2.rectangle(minimap, (0, 0), (200, 200), (20, 20, 20), -1)

    # Calculate arrow dimensions (target 18 pixels area)
    target_area = 18  # Middle of 12-25 range
    arrow_length = int(np.sqrt(target_area * 2))
    arrow_width = int(np.sqrt(target_area / 2))

    # Convert angle to drawing angle
    draw_angle = np.radians(arrow_angle)

    # Create arrow points
    tip_x = int(arrow_pos[0] + arrow_length * np.sin(draw_angle))
    tip_y = int(arrow_pos[1] - arrow_length * np.cos(draw_angle))

    base1_x = int(arrow_pos[0] + arrow_width * np.sin(draw_angle + 2.0944))
    base1_y = int(arrow_pos[1] - arrow_width * np.cos(draw_angle + 2.0944))

    base2_x = int(arrow_pos[0] + arrow_width * np.sin(draw_angle - 2.0944))
    base2_y = int(arrow_pos[1] - arrow_width * np.cos(draw_angle - 2.0944))

    pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw arrow (BGR values for yellow arrow)
    arrow_color = (20, 200, 230)  # BGR format matching yellow HSV range
    cv2.fillPoly(minimap, [pts], arrow_color)

    # Save debug visualizations
    cv2.imwrite('test_minimap_raw.png', minimap)

    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    cv2.imwrite('test_minimap_hsv.png', hsv)

    # Save arrow mask
    lower = np.array([20, 100, 200])  # Yellow arrow range
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite('test_minimap_mask.png', mask)

    return minimap

def test_map_loading():
    """Test map and resource data loading"""
    try:
        # Create test data
        test_map = create_test_map_data()

        # Initialize map manager
        map_manager = MapManager()

        # Test loading map
        success = map_manager.load_map('test_map')
        assert success, "Failed to load test map"

        # Test resource loading
        assert 'test_map' in map_manager.resource_nodes, "Resource nodes not loaded"
        assert len(map_manager.resource_nodes['test_map']) == 3, "Incorrect number of resource nodes"

        # Test resource retrieval
        nearby = map_manager.get_nearby_resources((100, 100), 60)
        assert len(nearby) == 3, "Failed to find nearby resources"

        # Test terrain detection
        terrain = map_manager.get_terrain_type((50, 50))
        assert terrain == 'forest', f"Incorrect terrain type: {terrain}"

        logger.info("✓ Map loading test passed")
        return True

    except Exception as e:
        logger.error(f"Map loading test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_arrow_detection(map_manager, position, angle):
    """Test arrow detection at specific position and angle"""
    minimap = create_test_minimap(position, angle)
    detected_pos = map_manager.detect_player_position(minimap)

    if detected_pos:
        logger.info(f"Test position {position}, angle {angle}°:")
        logger.info(f"Detected position: ({detected_pos.x}, {detected_pos.y})")
        logger.info(f"Detected angle: {detected_pos.direction:.1f}°")

        # Calculate error
        pos_error = np.sqrt((detected_pos.x - position[0])**2 + 
                         (detected_pos.y - position[1])**2)
        angle_error = min((detected_pos.direction - angle) % 360,
                       (angle - detected_pos.direction) % 360)

        logger.info(f"Position error: {pos_error:.1f} pixels")
        logger.info(f"Angle error: {angle_error:.1f}°")

        # Save visualization
        visualization = minimap.copy()
        cv2.circle(visualization, (detected_pos.x, detected_pos.y), 3, (0, 255, 0), -1)
        cv2.line(visualization, 
                (detected_pos.x, detected_pos.y),
                (int(detected_pos.x + 20 * np.sin(np.radians(detected_pos.direction))),
                 int(detected_pos.y - 20 * np.cos(np.radians(detected_pos.direction)))),
                (0, 255, 0), 2)
        cv2.imwrite(f'test_arrow_{angle}.png', visualization)

        return pos_error < 3 and angle_error < 15
    return False

def test_minimap_detection():
    """Test detection of player position and direction from minimap"""
    try:
        map_manager = MapManager()

        # First test resource loading
        if not test_map_loading():
            logger.error("Resource loading test failed")
            return False

        # Test cases with different positions and angles
        test_cases = [
            ((100, 100), 0),    # Center, pointing North
            ((50, 50), 90),     # Upper left, pointing East
            ((150, 150), 180),  # Lower right, pointing South
            ((150, 50), 270),   # Upper right, pointing West
            ((100, 100), 45),   # Center, pointing Northeast
        ]

        passed_tests = 0
        for position, angle in test_cases:
            if test_arrow_detection(map_manager, position, angle):
                passed_tests += 1
                logger.info(f"✓ Passed test: position {position}, angle {angle}°")
            else:
                logger.error(f"✗ Failed test: position {position}, angle {angle}°")

        success_rate = (passed_tests / len(test_cases)) * 100
        logger.info(f"\nTest Results: {passed_tests}/{len(test_cases)} passed ({success_rate:.1f}%)")

        return success_rate > 80  # Require at least 80% success rate

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_minimap_detection()
    if success:
        logger.info("Minimap detection test completed successfully")
    else:
        logger.error("Minimap detection test failed")
        exit(1)