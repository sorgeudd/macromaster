"""Test module for fishing bot functionality"""
import unittest
from mock_environment import create_test_environment
from bot_core import FishingBot
import logging
import time
import tempfile
import json
import os
import csv

class TestFishingBot(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.mock_env = create_test_environment()
        self.mock_env.start_simulation()
        self.bot = FishingBot(test_mode=True, test_env=self.mock_env)

    def tearDown(self):
        """Cleanup test environment"""
        try:
            if hasattr(self, 'bot'):
                if self.bot.running:
                    self.logger.info("Stopping bot in tearDown")
                    self.bot.stop()
                    if self.bot.bot_thread:
                        self.bot.bot_thread.join(timeout=2.0)
                        if self.bot.bot_thread.is_alive():
                            self.logger.warning("Bot thread still alive after timeout")
                if self.bot.learning_mode:
                    self.logger.info("Stopping learning mode in tearDown")
                    self.bot.stop_learning_mode()
            if hasattr(self, 'mock_env'):
                self.logger.info("Stopping mock environment in tearDown")
                self.mock_env.stop_simulation()
                # Wait for mock environment thread to finish
                if hasattr(self.mock_env, 'simulation_thread'):
                    self.mock_env.simulation_thread.join(timeout=2.0)
        except Exception as e:
            self.logger.error(f"Error in tearDown: {str(e)}")
            raise

    def test_learning_mode(self):
        """Test gameplay learning functionality"""
        # Start learning mode
        self.bot.start_learning_mode()
        self.assertTrue(self.bot.learning_mode)
        self.assertFalse(self.bot.adaptive_mode)

        # Simulate actions
        self.bot.record_action('move', (100, 200))
        self.bot.record_action('cast', (100, 200))
        self.bot.click()
        self.bot.press_key('f')

        # Wait for actions to be recorded
        time.sleep(0.5)

        # Stop learning mode
        self.bot.stop_learning_mode()
        self.assertFalse(self.bot.learning_mode)

        # Verify patterns were learned
        recorded_actions = self.bot.gameplay_learner.recorded_actions
        self.assertGreater(len(recorded_actions), 0)

        # Verify specific actions were recorded
        action_types = [a.action_type for a in recorded_actions]
        self.assertIn('move', action_types)
        self.assertIn('cast', action_types)

    def test_full_gameplay_cycle(self):
        """Test complete gameplay cycle with learning"""
        # Start learning mode
        self.bot.start_learning_mode()

        # Start bot
        self.bot.start()
        time.sleep(1)  # Allow initialization

        # Record initial events count
        initial_events = len(self.mock_env.input_events)

        # Simulate fish bite to trigger reeling
        self.mock_env.set_game_state(fish_bite_active=True)
        time.sleep(0.5)  # Wait for bite detection

        # Verify bite detection triggered reel action
        events_after_bite = [e for e in self.mock_env.input_events[initial_events:] 
                             if e['type'] == 'key_press' and e['key'] == 'r']
        self.assertGreater(len(events_after_bite), 0, "No reeling occurred after bite")

        # Clear bite state
        self.mock_env.set_game_state(fish_bite_active=False)
        time.sleep(0.5)

        # Stop bot and learning mode
        self.bot.stop()
        self.bot.stop_learning_mode()

        # Verify actions were recorded
        recorded_actions = self.bot.gameplay_learner.recorded_actions
        self.assertGreater(len(recorded_actions), 0, "No actions recorded in learning mode")

        # Switch to adaptive mode and verify behavior
        self.bot.start_adaptive_mode()
        self.assertTrue(self.bot.adaptive_mode)
        self.assertFalse(self.bot.learning_mode)

        # Let it run in adaptive mode
        self.bot.start()
        time.sleep(1)  # Allow adaptive mode to initialize

        # Simulate conditions that should trigger adaptive actions
        self.mock_env.set_game_state(fish_bite_active=True)
        time.sleep(1.0)  # Wait for adaptive response
        self.mock_env.set_game_state(fish_bite_active=False)
        time.sleep(1.0)  # Allow completion of action

        # Stop bot and verify adaptive actions occurred
        self.bot.stop()
        self.assertTrue(len(self.mock_env.input_events) > initial_events, 
                        "No adaptive actions occurred")

    def test_fish_detection(self):
        """Test fish bite detection"""
        # Simulate fish bite
        self.mock_env.set_game_state(fish_bite_active=True)
        self.assertTrue(self.bot._detect_bite())

        # Clear fish bite
        self.mock_env.set_game_state(fish_bite_active=False)
        self.assertFalse(self.bot._detect_bite())

    def test_combat_response(self):
        """Test combat detection and response"""
        # Simulate combat with health at 80%
        self.mock_env.set_game_state(is_in_combat=True, health=80.0)
        self.assertTrue(self.bot.check_combat_status())
        self.assertEqual(self.bot.get_current_health(), 80.0)

        # Set a timeout for combat handling
        start_time = time.time()
        max_combat_duration = 5.0  # Maximum time to wait for combat

        # Test combat handling
        self.bot._handle_combat()

        # Verify combat didn't take too long
        combat_duration = time.time() - start_time
        self.assertLess(combat_duration, max_combat_duration, 
                        f"Combat handling took too long: {combat_duration:.1f}s")

        # Verify combat actions were taken
        combat_events = [e for e in self.mock_env.input_events if e['type'] == 'key_press']
        self.assertGreater(len(combat_events), 0, "No combat abilities used")

        # Verify combat ended
        self.mock_env.set_game_state(is_in_combat=False)
        self.assertFalse(self.bot.check_combat_status(), "Combat state not cleared")

    def test_initialization(self):
        """Test bot initialization"""
        self.assertIsNotNone(self.bot)
        self.assertTrue(self.bot.test_mode)
        self.assertFalse(self.bot.running)

    def test_adaptive_mode(self):
        """Test adaptive gameplay based on learned patterns"""
        # First learn some patterns
        self.bot.start_learning_mode()
        self.bot.move_mouse_to(100, 200)
        self.bot.press_key('f')
        time.sleep(0.5)
        self.bot.stop_learning_mode()

        # Start adaptive mode
        self.bot.start_adaptive_mode()
        self.assertTrue(self.bot.adaptive_mode)
        self.assertFalse(self.bot.learning_mode)

        # Get next action recommendation
        next_action = self.bot.get_next_action()
        if next_action:
            self.assertIn('type', next_action)
            self.assertIn('timing', next_action)

    def test_obstacle_detection(self):
        """Test obstacle detection and avoidance"""
        obstacles = [(10, 10), (20, 20)]
        self.mock_env.set_game_state(detected_obstacles=obstacles)
        detected = self.bot.scan_for_obstacles()
        self.assertEqual(len(detected), 2)
        self.assertIn((10, 10), detected)

    def test_resource_detection(self):
        """Test resource detection"""
        resources = [{'type': 'herb', 'position': (50, 50)}]
        self.mock_env.set_game_state(detected_resources=resources)
        detected = self.bot.scan_for_resources()
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]['type'], 'herb')

    def test_input_control(self):
        """Test basic input controls"""
        # Test mouse movement
        result = self.bot.move_mouse_to(100, 200)
        self.assertTrue(result)
        last_event = self.mock_env.input_events[-1]
        self.assertEqual(last_event['type'], 'mouse_move')
        self.assertEqual(last_event['x'], 100)

        # Test key press
        result = self.bot.press_key('f')
        self.assertTrue(result)
        last_event = self.mock_env.input_events[-1]
        self.assertEqual(last_event['type'], 'key_press')
        self.assertEqual(last_event['key'], 'f')

    def test_window_management(self):
        """Test window detection and management"""
        # Test window finding
        success = self.bot.find_game_window("Test Game")
        self.assertTrue(success)
        self.assertIsNotNone(self.bot.window_handle)
        self.assertIsNotNone(self.bot.window_rect)

        # Test setting window region
        region = (0, 0, 800, 600)
        success = self.bot.set_window_region(region)
        self.assertTrue(success)
        self.assertEqual(self.bot.config['detection_area'], region)

        # Test window activation
        self.assertTrue(self.bot.is_window_active())
        self.assertTrue(self.bot.activate_window())

        # Test screenshot capture
        screenshot = self.bot.get_window_screenshot()
        self.assertIsNotNone(screenshot)

    def test_window_detection_comprehensive(self):
        """Test comprehensive window detection functionality"""
        # Test 1: Valid window title
        success, message = self.bot.find_game_window("Test Game")
        self.assertTrue(success)
        self.assertIn("Found window", message)

        # Verify window info is populated
        window_info = self.bot.get_window_info()
        self.assertIsNotNone(window_info)
        self.assertEqual(window_info['rect'], (0, 0, 800, 600))

        # Test 2: Empty window title
        success, message = self.bot.find_game_window("")
        self.assertFalse(success)
        self.assertIn("provide a window title", message)

        # Test 3: Invalid window title
        success, message = self.bot.find_game_window("NonexistentWindow12345")
        self.assertFalse(success)
        self.assertIn("not found", message)

        # Test 4: Notepad (valid test window)
        success, message = self.bot.find_game_window("Notepad")
        self.assertTrue(success)
        self.assertIn("Found window", message)


    def test_map_functionality(self):
        """Test map loading and navigation features"""
        # Create a test map file
        test_map = {
            'nodes': [
                {'id': 1, 'x': 100, 'y': 100, 'type': 'resource', 'resource': 'fish'},
                {'id': 2, 'x': 200, 'y': 200, 'type': 'resource', 'resource': 'ore'},
                {'id': 3, 'x': 150, 'y': 150, 'type': 'path'}
            ],
            'edges': [
                {'from': 1, 'to': 3},
                {'from': 3, 'to': 2}
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_map, tmp)
            tmp_path = tmp.name

        try:
            # Test map validation
            self.assertTrue(self.bot._validate_map_data(test_map))

            # Test loading from file
            success = self.bot.load_map_data(tmp_path)
            self.assertTrue(success)

            # Test navigation with loaded map
            self.bot.navigate_to((200, 200))  # Should use loaded map data

            # Verify navigation events
            nav_events = [e for e in self.mock_env.input_events 
                         if e['type'] in ('mouse_move', 'key_press')]
            self.assertGreater(len(nav_events), 0)

            # Test invalid map data
            invalid_map = {'nodes': []}  # Missing required data
            self.assertFalse(self.bot._validate_map_data(invalid_map))

            # Test CSV format
            csv_data = [
                {'x': '100', 'y': '100', 'type': 'resource'},
                {'x': '200', 'y': '200', 'type': 'path'}
            ]
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
                writer = csv.DictWriter(tmp_csv, fieldnames=['x', 'y', 'type'])
                writer.writeheader()
                writer.writerows(csv_data)
                tmp_csv_path = tmp_csv.name

            success = self.bot.load_map_data(tmp_csv_path)
            self.assertTrue(success)
            os.unlink(tmp_csv_path)

        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()