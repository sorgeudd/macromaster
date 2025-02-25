"""Test script for game window movement and coordinate translation"""
import cv2
import numpy as np
import logging
from pathlib import Path
from game_window import GameWindow
from map_manager import MapManager

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('GameWindowTest')

def test_coordinate_translation():
    """Test coordinate translation between game window and world space"""
    try:
        game_window = GameWindow()
        
        # Test various positions and camera rotations
        test_cases = [
            ((100, 100), (0, 0), 0),    # Center position, no rotation
            ((200, 150), (50, 30), 45),  # Offset position, 45-degree rotation
            ((50, 200), (-30, 40), 90),  # Negative offset, 90-degree rotation
            ((300, 300), (100, 100), 180)  # Far position, 180-degree rotation
        ]
        
        passed_tests = 0
        for world_pos, player_pos, rotation in test_cases:
            # Set camera rotation
            game_window.camera_rotation = rotation
            
            # Convert world to screen coordinates
            screen_pos = game_window.translate_world_to_screen(world_pos, player_pos)
            
            # Check if position is visible
            is_visible = game_window.is_position_visible(world_pos, player_pos)
            
            # Test movement vector calculation
            move_vector = game_window.get_movement_vector(player_pos, world_pos)
            
            # Log results
            logger.info(f"\nTest case: world_pos={world_pos}, player_pos={player_pos}, rotation={rotation}°")
            logger.info(f"Screen position: {screen_pos}")
            logger.info(f"Visible: {is_visible}")
            logger.info(f"Movement vector: ({move_vector[0]:.2f}, {move_vector[1]:.2f})")
            
            # Verify reasonable results
            if (0 <= screen_pos[0] <= game_window.view_width and 
                0 <= screen_pos[1] <= game_window.view_height):
                passed_tests += 1
                logger.info("✓ Position translation test passed")
            else:
                logger.error("✗ Position translation test failed")
        
        # Test screen movement calculation
        delta_time = 0.016  # Simulate 60 FPS
        movement = game_window.calculate_screen_movement(
            (0, 0),         # Current position
            (100, 100),     # Target position
            delta_time
        )
        logger.info(f"\nScreen movement: ({movement[0]:.2f}, {movement[1]:.2f})")
        
        # Test direction vector calculation
        for angle in [0, 45, 90, 180, 270]:
            direction = game_window.get_screen_direction(angle)
            logger.info(f"Screen direction at {angle}°: ({direction[0]:.2f}, {direction[1]:.2f})")
        
        success_rate = (passed_tests / len(test_cases)) * 100
        logger.info(f"\nTest Results: {passed_tests}/{len(test_cases)} passed ({success_rate:.1f}%)")
        
        return success_rate >= 80  # Require at least 80% success rate
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_movement_integration():
    """Test integration between map manager and game window movement"""
    try:
        map_manager = MapManager()
        game_window = GameWindow()
        
        # Create test minimap
        minimap_size = (200, 200, 3)
        minimap = np.zeros(minimap_size, dtype=np.uint8)
        
        # Add test arrow
        cv2.circle(minimap, (100, 100), 5, (200, 150, 100), -1)  # Arrow center
        cv2.line(minimap, (100, 100), (100, 80), (200, 150, 100), 2)  # Arrow direction
        
        # Detect player position from minimap
        position = map_manager.detect_player_position(minimap)
        if not position:
            logger.error("Failed to detect player position")
            return False
            
        # Test world to screen coordinate conversion
        world_pos = map_manager.minimap_to_world_coords((position.x, position.y))
        screen_pos = game_window.translate_world_to_screen(
            world_pos,
            (game_window.view_width//2, game_window.view_height//2)
        )
        
        logger.info(f"\nIntegration test results:")
        logger.info(f"Minimap position: ({position.x}, {position.y})")
        logger.info(f"World position: ({world_pos[0]}, {world_pos[1]})")
        logger.info(f"Screen position: ({screen_pos[0]}, {screen_pos[1]})")
        
        # Test movement calculation
        movement = game_window.calculate_screen_movement(
            (game_window.view_width//2, game_window.view_height//2),  # Current center
            screen_pos,  # Target position
            0.016  # Delta time
        )
        
        logger.info(f"Movement vector: ({movement[0]:.2f}, {movement[1]:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    coord_test = test_coordinate_translation()
    integration_test = test_movement_integration()
    
    if coord_test and integration_test:
        logger.info("\nAll game window tests completed successfully")
    else:
        logger.error("\nGame window tests failed")
        exit(1)
