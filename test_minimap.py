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
    # Create black background with consistent size
    minimap_size = (200, 200, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Create darker background for better contrast
    cv2.rectangle(minimap, (0, 0), (200, 200), (20, 20, 20), -1)

    # Calculate arrow dimensions to match target area (12-25 pixels)
    target_area = 18  # Target middle of our range
    arrow_length = int(np.sqrt(target_area * 2))  # Length ~sqrt(2) times width
    arrow_width = int(np.sqrt(target_area / 2))

    # Convert game angle to drawing angle (0° is up, increases clockwise)
    draw_angle = np.radians(arrow_angle)

    # Create arrow shape with specific aspect ratio (2:1)
    tip_x = int(arrow_pos[0] + arrow_length * np.sin(draw_angle))
    tip_y = int(arrow_pos[1] - arrow_length * np.cos(draw_angle))

    base1_x = int(arrow_pos[0] + arrow_width * np.sin(draw_angle + 2.0944))  # +120°
    base1_y = int(arrow_pos[1] - arrow_width * np.cos(draw_angle + 2.0944))

    base2_x = int(arrow_pos[0] + arrow_width * np.sin(draw_angle - 2.0944))  # -120°
    base2_y = int(arrow_pos[1] - arrow_width * np.cos(draw_angle - 2.0944))

    # Create arrow points
    pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw arrow with exact HSV color that matches detection thresholds
    arrow_color = (105, 185, 210)  # BGR format matching HSV thresholds
    cv2.fillPoly(minimap, [pts], arrow_color)

    # Save intermediate results for debugging
    cv2.imwrite('test_minimap_raw.png', minimap)

    # Debug visualization in HSV
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    cv2.imwrite('test_minimap_hsv.png', hsv)

    # Save arrow mask for debugging
    lower = np.array([95, 170, 200])
    upper = np.array([105, 190, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite('test_minimap_mask.png', mask)

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