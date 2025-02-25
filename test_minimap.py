"""Test script for minimap player position detection and resource tracking"""
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from map_manager import MapManager, ResourceNode
import shutil
import traceback

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('MinimapTest')

def setup_test_maps():
    """Setup test maps directory with actual fishing spot map"""
    test_map_dir = Path('maps')
    test_map_dir.mkdir(exist_ok=True)

    # Copy the actual resource maps
    for resource_map in [
        ('attached_assets/Mase knoll fish_1740481898173.jpg', 'mase_knoll_fish.png'),
        ('attached_assets/image_1740481868399.png', 'mase_knoll_ore.png')
    ]:
        source, target = resource_map
        shutil.copy(source, test_map_dir / target)

def test_fishing_spot_detection():
    """Test detection of fishing spots from map"""
    try:
        # Setup test environment
        setup_test_maps()

        # Initialize map manager
        map_manager = MapManager()

        # Load the fishing spot map
        success = map_manager.load_map('mase knoll fish')
        assert success, "Failed to load fishing map"

        # Verify fishing spots were detected
        assert 'mase_knoll_fish' in map_manager.resource_nodes, "No resource nodes found"
        spots = map_manager.resource_nodes['mase_knoll_fish']
        assert len(spots) > 0, "No fishing spots detected"

        logger.info(f"Detected {len(spots)} fishing spots")

        # Test nearby spot detection
        for spot in spots[:3]:  # Log first 3 spots
            logger.debug(f"Fishing spot at {spot.position}")
            nearby = map_manager.get_nearby_resources(spot.position, 50)
            assert len(nearby) > 0, "Failed to find nearby spots"

        logger.info("✓ Fishing spot detection test passed")
        return True

    except Exception as e:
        logger.error(f"Fishing spot detection test failed: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

def create_test_minimap(arrow_pos=(100, 100), angle=0, zone_name="Mase Knoll"):
    """Create a test minimap with player arrow"""
    # Create black background
    minimap_size = (200, 200, 3)
    minimap = np.zeros(minimap_size, dtype=np.uint8)

    # Create darker background for better contrast
    cv2.rectangle(minimap, (0, 0), (200, 200), (20, 20, 20), -1)

    # Calculate arrow dimensions
    target_area = 25  # Target arrow size
    arrow_length = int(np.sqrt(target_area * 2))
    arrow_width = int(np.sqrt(target_area / 2))

    # Create arrow points
    tip_x = int(arrow_pos[0] + arrow_length * np.sin(np.radians(angle)))
    tip_y = int(arrow_pos[1] - arrow_length * np.cos(np.radians(angle)))

    base1_x = int(arrow_pos[0] + arrow_width * np.sin(np.radians(angle + 120)))
    base1_y = int(arrow_pos[1] - arrow_width * np.cos(np.radians(angle + 120)))

    base2_x = int(arrow_pos[0] + arrow_width * np.sin(np.radians(angle - 120)))
    base2_y = int(arrow_pos[1] - arrow_width * np.cos(np.radians(angle - 120)))

    pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw arrow
    arrow_color = (20, 200, 230)  # BGR format for yellow arrow
    cv2.fillPoly(minimap, [pts], arrow_color)
    cv2.polylines(minimap, [pts], True, arrow_color, 1)

    # Add zone name with white color for better visibility
    cv2.putText(minimap, zone_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return minimap

def test_arrow_detection(map_manager, position, angle=0):
    """Test arrow detection at specific position"""
    minimap = create_test_minimap(position)
    detected_pos = map_manager.detect_player_position(minimap)

    if detected_pos:
        logger.info(f"Test position {position}:")
        logger.info(f"Detected position: ({detected_pos.x}, {detected_pos.y})")

        # Calculate position error
        pos_error = np.sqrt((detected_pos.x - position[0])**2 + 
                         (detected_pos.y - position[1])**2)

        logger.info(f"Position error: {pos_error:.1f} pixels")

        return pos_error < 3  # Allow up to 3 pixels error

    return False

def test_minimap_detection():
    """Test detection of player position from minimap"""
    try:
        map_manager = MapManager()

        # Test cases with different positions
        test_positions = [
            (100, 100),  # Center
            (50, 50),    # Upper left
            (150, 150),  # Lower right
            (150, 50),   # Upper right
            (100, 100)   # Center again
        ]

        passed_tests = 0
        for position in test_positions:
            if test_arrow_detection(map_manager, position):
                passed_tests += 1
                logger.info(f"✓ Passed test: position {position}")
            else:
                logger.error(f"✗ Failed test: position {position}")

        success_rate = (passed_tests / len(test_positions)) * 100
        logger.info(f"\nTest Results: {passed_tests}/{len(test_positions)} passed ({success_rate:.1f}%)")

        return success_rate > 80  # Require at least 80% success rate

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Test fishing spot detection first
    if not test_fishing_spot_detection():
        logger.error("Fishing spot detection test failed")
        exit(1)

    # Test minimap detection
    if not test_minimap_detection():
        logger.error("Minimap detection test failed")
        exit(1)

    logger.info("All tests completed successfully")