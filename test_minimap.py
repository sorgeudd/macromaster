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
        arrow_angle: angle in degrees (0 is North, 90 is East)
    """
    minimap_size = (200, 200, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Add varied terrain features
    # Dark terrain area
    cv2.rectangle(minimap, (50, 50), (150, 150), (60, 60, 60), -1)
    # Mountain region
    cv2.circle(minimap, (120, 80), 30, (120, 80, 40), -1)
    # Deep water
    cv2.rectangle(minimap, (20, 150), (70, 180), (150, 80, 40), -1)
    # Shallow water
    cv2.rectangle(minimap, (80, 150), (130, 180), (200, 120, 60), -1)

    # Calculate arrow points based on angle
    arrow_length = 20
    arrow_width = 10
    angle_rad = np.radians(arrow_angle - 90)  # Adjust angle to match game coordinates

    # Calculate arrow tip and base points
    tip_x = arrow_pos[0] + arrow_length * np.cos(angle_rad)
    tip_y = arrow_pos[1] + arrow_length * np.sin(angle_rad)

    base_angle1 = angle_rad + np.pi * 2/3
    base_angle2 = angle_rad - np.pi * 2/3

    base1_x = arrow_pos[0] + arrow_width * np.cos(base_angle1)
    base1_y = arrow_pos[1] + arrow_width * np.sin(base_angle1)

    base2_x = arrow_pos[0] + arrow_width * np.cos(base_angle2)
    base2_y = arrow_pos[1] + arrow_width * np.sin(base_angle2)

    # Draw player arrow
    arrow_points = np.array([
        [tip_x, tip_y],         # Tip
        [base1_x, base1_y],     # Left base
        [base2_x, base2_y]      # Right base
    ], np.int32)

    # Fill arrow in light blue (BGR format)
    cv2.fillPoly(minimap, [arrow_points], (200, 150, 100))

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

        # Save visualization of last test case
        minimap = create_test_minimap(*test_cases[-1])
        detected_pos = map_manager.detect_player_position(minimap)

        if detected_pos:
            visualization = minimap.copy()

            # Draw detected position
            cv2.circle(visualization, (detected_pos.x, detected_pos.y), 
                      5, (0, 255, 0), -1)

            # Draw detected direction
            direction = map_manager.get_player_direction_vector()
            if direction:
                end_x = int(detected_pos.x + direction[0] * 30)
                end_y = int(detected_pos.y + direction[1] * 30)
                cv2.line(visualization, 
                        (detected_pos.x, detected_pos.y),
                        (end_x, end_y), 
                        (0, 255, 0), 2)

            # Add test information
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(visualization, 
                       f"Direction: {detected_pos.direction:.1f}°",
                       (10, 20), font, 0.5, (255, 255, 255), 1)

            cv2.imwrite('minimap_detection.png', visualization)
            logger.info("Saved detection visualization to minimap_detection.png")

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