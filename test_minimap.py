"""Test script for minimap player position detection"""
import cv2
import numpy as np
import logging
from pathlib import Path
from map_manager import MapManager

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MinimapTest')

def create_test_minimap(arrow_pos=(100, 100), arrow_angle=0):
    """Create a test minimap with player arrow at specified position and angle
    Args:
        arrow_pos: (x, y) position of arrow center
        arrow_angle: angle in degrees (0 is North, clockwise)
    """
    # Create black background - sized for test environment
    minimap_size = (250, 250, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Add varied terrain features
    cv2.rectangle(minimap, (50, 50), (150, 150), (40, 40, 40), -1)
    cv2.circle(minimap, (120, 80), 30, (80, 60, 40), -1)
    cv2.rectangle(minimap, (20, 150), (70, 180), (120, 60, 20), -1)
    cv2.rectangle(minimap, (80, 150), (130, 180), (160, 100, 40), -1)

    # Arrow dimensions (scaled down from high-res footage proportions)
    arrow_length = 3  # Scaled from 14
    arrow_width = 5   # Scaled from 27

    # Convert game angle to drawing angle (0° is up, increases clockwise)
    draw_angle = np.radians(arrow_angle)

    # Create arrow points
    arrow_points = []

    # Tip point
    tip_x = int(arrow_pos[0] + arrow_length * np.sin(draw_angle))
    tip_y = int(arrow_pos[1] - arrow_length * np.cos(draw_angle))
    arrow_points.append([tip_x, tip_y])

    # Base points - create triangular shape
    for offset in [-2.0944, 2.0944]:  # ±120 degrees
        base_angle = draw_angle + offset
        base_x = int(arrow_pos[0] + arrow_width/2 * np.sin(base_angle))
        base_y = int(arrow_pos[1] - arrow_width/2 * np.cos(base_angle))
        arrow_points.append([base_x, base_y])

    # Convert to numpy array and reshape for fillPoly
    pts = np.array(arrow_points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Fill arrow with exact game color
    cv2.fillPoly(minimap, [pts], (105, 210, 210))  # BGR format from footage

    return minimap

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