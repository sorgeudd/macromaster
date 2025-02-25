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
        arrow_pos: (x, y) position of arrow
        arrow_angle: angle in degrees (0 is North, clockwise)
    """
    minimap_size = (200, 200, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Add varied terrain features - matched to game colors
    cv2.rectangle(minimap, (50, 50), (150, 150), (40, 40, 40), -1)
    cv2.circle(minimap, (120, 80), 30, (80, 60, 40), -1)
    cv2.rectangle(minimap, (20, 150), (70, 180), (120, 60, 20), -1)
    cv2.rectangle(minimap, (80, 150), (130, 180), (160, 100, 40), -1)

    # Calculate arrow points using game coordinates
    arrow_length = 12  # Matched to game size
    arrow_width = 6   # Matched to game width

    # Convert angle to radians (0° is North, increases clockwise)
    angle_rad = np.radians(arrow_angle)

    # Calculate arrow points
    # Tip points upward at 0°, rotates clockwise
    tip_x = arrow_pos[0] + arrow_length * np.sin(angle_rad)
    tip_y = arrow_pos[1] - arrow_length * np.cos(angle_rad)

    # Calculate base points
    base_angle1 = angle_rad + np.pi * 5/6
    base_angle2 = angle_rad - np.pi * 5/6

    base1_x = arrow_pos[0] + arrow_width * np.sin(base_angle1)
    base1_y = arrow_pos[1] - arrow_width * np.cos(base_angle1)

    base2_x = arrow_pos[0] + arrow_width * np.sin(base_angle2)
    base2_y = arrow_pos[1] - arrow_width * np.cos(base_angle2)

    # Create arrow polygon
    arrow_points = np.array([
        [tip_x, tip_y],      # Tip
        [base1_x, base1_y],  # Base left
        [base2_x, base2_y]   # Base right
    ], np.int32)

    # Fill arrow with game-accurate color
    cv2.fillPoly(minimap, [arrow_points], (100, 180, 220))

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

        # Save visualization for debugging
        visualization = minimap.copy()
        cv2.circle(visualization, (detected_pos.x, detected_pos.y), 3, (0, 255, 0), -1)
        cv2.line(visualization,
                (detected_pos.x, detected_pos.y),
                (int(detected_pos.x + 20 * np.sin(np.radians(detected_pos.direction))),
                 int(detected_pos.y - 20 * np.cos(np.radians(detected_pos.direction)))),
                (0, 255, 0), 2)
        cv2.imwrite(f'test_arrow_{angle}.png', visualization)

        return pos_error < 5 and angle_error < 10
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