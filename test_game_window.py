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

def create_test_visualization(game_window, screen_pos):
    """Create visualization for coordinate system testing"""
    viz_image = np.zeros((game_window.window_height, game_window.window_width, 3), dtype=np.uint8)

    # Draw coordinate grid
    grid_spacing = 50
    grid_color = (30, 30, 30)
    for x in range(0, game_window.window_width, grid_spacing):
        cv2.line(viz_image, (x, 0), (x, game_window.window_height), grid_color, 1)
    for y in range(0, game_window.window_height, grid_spacing):
        cv2.line(viz_image, (0, y), (game_window.window_width, y), grid_color, 1)

    # Draw main axes
    cv2.line(viz_image, 
            (game_window.window_width//2, 0), 
            (game_window.window_width//2, game_window.window_height), 
            (50, 50, 50), 2)
    cv2.line(viz_image,
            (0, game_window.window_height//2),
            (game_window.window_width, game_window.window_height//2),
            (50, 50, 50), 2)

    # Draw viewport boundaries
    viewport_color = (0, 100, 200)
    cv2.rectangle(viz_image,
                (game_window.viewport_offset_x, game_window.viewport_offset_y),
                (game_window.viewport_offset_x + game_window.view_width, 
                 game_window.viewport_offset_y + game_window.view_height),
                viewport_color, 2)

    # Draw screen position
    if screen_pos:
        cv2.circle(viz_image, 
                  (int(screen_pos[0]), int(screen_pos[1])), 
                  5, (0, 255, 0), -1)

        # Draw movement vector from center
        cv2.arrowedLine(viz_image,
                       (game_window.window_width//2, game_window.window_height//2),
                       (int(screen_pos[0]), int(screen_pos[1])),
                       (0, 255, 0), 2)

    # Add resolution info
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz_image, 
               f"Window: {game_window.window_width}x{game_window.window_height}", 
               (10, 30), font, 0.5, (255, 255, 255), 1)
    cv2.putText(viz_image,
               f"Viewport offset: ({game_window.viewport_offset_x}, {game_window.viewport_offset_y})",
               (10, 50), font, 0.5, (255, 255, 255), 1)

    return viz_image

def test_coordinate_translation():
    """Test coordinate translation between game window and world space"""
    try:
        game_window = GameWindow()

        # Test resolution configuration
        logger.info("Testing resolution configuration:")

        # Test default resolution
        logger.info("Default resolution:")
        logger.info(f"System: {game_window.system_width}x{game_window.system_height}")
        logger.info(f"Window: {game_window.window_width}x{game_window.window_height}")

        # Test custom resolution
        custom_res = (1280, 720)
        system_res = (1920, 1200)
        success = game_window.configure_resolution(
            custom_res[0], custom_res[1],
            system_res[0], system_res[1]
        )

        if success:
            logger.info("\nConfigured custom resolution:")
            logger.info(f"System: {game_window.system_width}x{game_window.system_height}")
            logger.info(f"Window: {game_window.window_width}x{game_window.window_height}")

        # Test viewport offset configurations
        viewport_tests = [
            (0, 0),      # Centered viewport
            (50, -30),   # Offset viewport
            (-20, 40),   # Negative offset
            (100, 100)   # Large offset
        ]

        passed_viewport_tests = 0
        for offset_x, offset_y in viewport_tests:
            game_window.viewport_offset_x = offset_x
            game_window.viewport_offset_y = offset_y

            # Create visualization for this viewport configuration
            test_screen_pos = (game_window.window_width//2 + 100, game_window.window_height//2 + 100)
            viz_image = create_test_visualization(game_window, test_screen_pos)

            # Test coordinate translations
            window_pos = game_window.translate_screen_to_window(test_screen_pos)
            screen_pos = game_window.translate_window_to_screen(window_pos)

            # Log results
            logger.info(f"\nViewport test with offset ({offset_x}, {offset_y}):")
            logger.info(f"Original screen pos: {test_screen_pos}")
            logger.info(f"Window pos: {window_pos}")
            logger.info(f"Translated screen pos: {screen_pos}")

            # Verify translations are within bounds
            if (0 <= window_pos[0] <= game_window.window_width and 
                0 <= window_pos[1] <= game_window.window_height):
                passed_viewport_tests += 1
                logger.info("✓ Viewport translation test passed")
            else:
                logger.error("✗ Viewport translation test failed")

            # Save visualization for this configuration
            cv2.imwrite(f'coordinate_translation_viewport_{offset_x}_{offset_y}.png', viz_image)

        viewport_success_rate = (passed_viewport_tests / len(viewport_tests)) * 100
        logger.info(f"\nViewport Test Results: {passed_viewport_tests}/{len(viewport_tests)} "
                   f"passed ({viewport_success_rate:.1f}%)")

        return viewport_success_rate >= 80

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

        # Configure game resolution
        game_window.configure_resolution(1280, 720, 1920, 1200)

        # Create test minimap
        minimap_size = (200, 200, 3)
        minimap = np.zeros(minimap_size, dtype=np.uint8)

        # Add test arrow (light blue color matching map_manager detection parameters)
        arrow_center = (100, 100)
        arrow_points = np.array([
            [100, 80],   # Tip (pointing North)
            [90, 110],   # Left base
            [110, 110]   # Right base
        ], np.int32)
        cv2.fillPoly(minimap, [arrow_points], (200, 150, 100))  # BGR format

        # Save test minimap for debugging
        cv2.imwrite('test_minimap.png', minimap)
        logger.info("Saved test minimap image")

        # Detect player position from minimap
        position = map_manager.detect_player_position(minimap)
        if not position:
            logger.error("Failed to detect player position")
            return False

        # Test world to screen coordinate conversion with different viewport offsets
        viewport_offsets = [(0, 0), (50, -30), (-20, 40)]
        for offset_x, offset_y in viewport_offsets:
            game_window.viewport_offset_x = offset_x
            game_window.viewport_offset_y = offset_y

            world_pos = map_manager.minimap_to_world_coords((position.x, position.y))
            screen_pos = game_window.translate_world_to_screen(
                world_pos,
                (game_window.view_width//2, game_window.view_height//2)
            )

            logger.info(f"\nIntegration test with viewport ({offset_x}, {offset_y}):")
            logger.info(f"Minimap position: ({position.x}, {position.y})")
            logger.info(f"World position: ({world_pos[0]}, {world_pos[1]})")
            logger.info(f"Screen position: ({screen_pos[0]}, {screen_pos[1]})")

            # Create and save visualization
            viz_image = create_test_visualization(game_window, screen_pos)
            cv2.imwrite(f'integration_test_viewport_{offset_x}_{offset_y}.png', viz_image)

        return True

    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_viewport_configuration():
    """Test viewport offset configuration and bounds checking"""
    try:
        game_window = GameWindow()

        # Configure with user's resolution
        game_window.configure_resolution(1280, 720, 1920, 1200)

        # Test viewport offset bounds
        test_offsets = [
            (0, 0),       # Center
            (100, 100),   # Positive offset
            (-100, -100), # Negative offset
            (2000, 2000), # Out of bounds (should be clamped)
        ]

        for offset_x, offset_y in test_offsets:
            game_window.set_viewport_offset(offset_x, offset_y)
            logger.info(f"\nTesting viewport offset ({offset_x}, {offset_y}):")
            logger.info(f"Actual offset: ({game_window.viewport_offset_x}, {game_window.viewport_offset_y})")

            # Test coordinate translation with this offset
            test_pos = (game_window.window_width//2, game_window.window_height//2)
            window_pos = game_window.translate_screen_to_window(test_pos)
            screen_pos = game_window.translate_window_to_screen(window_pos)

            # Create visualization
            viz = create_test_visualization(game_window, screen_pos)
            cv2.imwrite(f'viewport_test_{offset_x}_{offset_y}.png', viz)

        return True

    except Exception as e:
        logger.error(f"Viewport test failed: {str(e)}")
        return False

def test_view_size_update():
    """Test view size updates and dynamic resolution changes"""
    try:
        game_window = GameWindow()

        # Configure with user's resolution
        game_window.configure_resolution(1280, 720, 1920, 1200)

        # Test different view sizes
        test_sizes = [
            (1000, 600),   # Smaller view
            (1280, 720),   # Full window size
            (1400, 800),   # Larger than window (should be clamped)
        ]

        for view_width, view_height in test_sizes:
            # Update view size
            game_window.update_view_size(view_width, view_height)
            logger.info(f"\nTesting view size ({view_width}, {view_height}):")
            logger.info(f"Actual view size: {game_window.view_width}x{game_window.view_height}")

            # Test coordinate translation with new view size
            test_pos = (game_window.window_width//2, game_window.window_height//2)
            window_pos = game_window.translate_screen_to_window(test_pos)
            screen_pos = game_window.translate_window_to_screen(window_pos)

            # Create visualization
            viz = create_test_visualization(game_window, screen_pos)
            cv2.imwrite(f'view_size_test_{view_width}_{view_height}.png', viz)

        return True

    except Exception as e:
        logger.error(f"View size test failed: {str(e)}")
        return False

def test_screen_metrics_update():
    """Test dynamic screen metrics updates"""
    try:
        game_window = GameWindow()

        # Configure initial resolution
        game_window.configure_resolution(1280, 720, 1920, 1200)

        # Test different system resolutions
        test_resolutions = [
            (1920, 1200),  # Original resolution
            (2560, 1440),  # Higher resolution
            (1680, 1050),  # Different aspect ratio
        ]

        for sys_width, sys_height in test_resolutions:
            # Update system metrics
            success = game_window.update_screen_metrics(sys_width, sys_height)
            if not success:
                logger.error(f"Failed to update screen metrics to {sys_width}x{sys_height}")
                continue

            logger.info(f"\nTesting system resolution {sys_width}x{sys_height}:")
            logger.info(f"Window position: ({game_window.window_x}, {game_window.window_y})")

            # Test coordinate translation with new metrics
            test_pos = (sys_width//2, sys_height//2)  # Center of screen
            window_pos = game_window.translate_screen_to_window(test_pos)
            screen_pos = game_window.translate_window_to_screen(window_pos)

            # Create visualization
            viz = create_test_visualization(game_window, window_pos)
            cv2.putText(viz, 
                       f"System: {sys_width}x{sys_height}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(f'screen_metrics_test_{sys_width}x{sys_height}.png', viz)

        return True

    except Exception as e:
        logger.error(f"Screen metrics test failed: {str(e)}")
        return False

if __name__ == "__main__":
    coord_test = test_coordinate_translation()
    viewport_test = test_viewport_configuration()
    view_size_test = test_view_size_update()
    screen_metrics_test = test_screen_metrics_update()
    integration_test = test_movement_integration()

    if (coord_test and viewport_test and view_size_test and 
        screen_metrics_test and integration_test):
        logger.info("\nAll game window tests completed successfully")
    else:
        logger.error("\nGame window tests failed")
        exit(1)