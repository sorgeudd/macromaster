"""Mock environment for headless testing of fishing bot functionality"""
import logging
import traceback
from dataclasses import dataclass
from threading import Thread, Event
import time
import random
import numpy as np

@dataclass
class GameState:
    """Represents the current state of the game"""
    health: float = 100.0
    is_mounted: bool = False
    is_in_combat: bool = False
    current_position: tuple = (0, 0)
    detected_resources: list = None
    detected_obstacles: list = None
    fish_bite_active: bool = False
    window_active: bool = True
    screen_content: np.ndarray = None
    combat_start_time: float = None
    last_action_time: float = None
    ability_cooldowns: dict = None
    terrain_type: str = 'normal'
    terrain_speed_multiplier: float = 1.0

    def __post_init__(self):
        if self.detected_resources is None:
            self.detected_resources = []
        if self.detected_obstacles is None:
            self.detected_obstacles = []
        if self.screen_content is None:
            self.screen_content = np.zeros((600, 800, 3), dtype=np.uint8)
        if self.ability_cooldowns is None:
            self.ability_cooldowns = {
                'e': 0.0,
                'w': 0.0,
                'q': 0.0,
                'space': 0.0
            }
        self.last_action_time = time.time()
        self.combat_start_time = time.time()

class MockEnvironment:
    def __init__(self):
        # Initialize logging
        self.logger = logging.getLogger('MockEnvironment')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if not self.logger.handlers:
            file_handler = logging.FileHandler('mock_environment.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        # Initialize state
        self.state = GameState()
        self.input_events = []
        self.fish_bite_event = Event()
        self.running = False
        self.min_action_interval = 0.05

        # Terrain-specific settings
        self.terrain_speeds = {
            'normal': 1.0,
            'water': 0.7,
            'mountain': 0.5,
            'forest': 0.8
        }

        # Initialize mock screen
        self.window_size = (800, 600)
        self.state.screen_content = np.zeros((*self.window_size, 3), dtype=np.uint8)

        self.logger.info("MockEnvironment initialized successfully")

    def move_mouse(self, x, y):
        """Record mouse movement and update position"""
        try:
            self.logger.debug(f"Moving mouse to ({x}, {y})")
            self.state.current_position = (x, y)
            success = self.record_input('mouse_move', x=x, y=y)

            # Add terrain-based delay
            speed_mult = self.terrain_speeds.get(self.state.terrain_type, 1.0)
            move_delay = 0.05 / speed_mult
            time.sleep(move_delay * 0.1)  # Scale down for tests

            return success

        except Exception as e:
            self.logger.error(f"Error in move_mouse: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def click(self, button='left', clicks=1):
        """Record mouse click with terrain-based delays"""
        try:
            self.logger.debug(f"Processing click: button={button}, clicks={clicks}")
            pos = self.state.current_position

            # Record click event at current position
            success = self.record_input('mouse_click', 
                                      button=button, 
                                      clicks=clicks,
                                      x=pos[0],
                                      y=pos[1])

            # Add terrain-based delay
            speed_mult = self.terrain_speeds.get(self.state.terrain_type, 1.0)
            click_delay = 0.1 / speed_mult
            time.sleep(click_delay * 0.1)  # Scale down for tests

            self.logger.debug(f"Click recorded at {pos}")
            return success

        except Exception as e:
            self.logger.error(f"Error recording click: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def move_to(self, position):
        """Move to position with terrain-aware navigation"""
        try:
            start_pos = self.state.current_position
            if not start_pos:
                start_pos = (0, 0)

            # Calculate distance
            dx = position[0] - start_pos[0]
            dy = position[1] - start_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)

            # Move mouse to new position
            success = self.move_mouse(position[0], position[1])
            if not success:
                return False

            # Click at destination
            success = self.click(button='left')
            if not success:
                return False

            # Additional delay based on distance and terrain
            speed_mult = self.terrain_speeds.get(self.state.terrain_type, 1.0)
            move_time = distance * 0.001 / speed_mult
            time.sleep(move_time * 0.1)  # Scale down for tests

            self.logger.debug(f"Moved to {position} (terrain: {self.state.terrain_type})")
            return True

        except Exception as e:
            self.logger.error(f"Error in move_to: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def record_input(self, input_type, **kwargs):
        """Record input events with terrain information"""
        try:
            event = {
                'type': input_type,
                'timestamp': time.time(),
                'terrain_type': self.state.terrain_type,
                **kwargs
            }

            self.input_events.append(event)
            self.logger.debug(f"Recorded input event: {event}")
            return True

        except Exception as e:
            self.logger.error(f"Error recording input: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def press_key(self, key, duration=None):
        """Record key press with detailed logging"""
        try:
            self.logger.debug(f"Processing key press: {key} (duration: {duration})")

            # Record key press
            success = self.record_input('key_press', key=key, duration=duration)

            # Handle special keys
            if key == 'a':
                self.state.is_mounted = not self.state.is_mounted
            elif key == 'r' and self.state.fish_bite_active:
                self.fish_bite_event.clear()
            elif key in self.state.ability_cooldowns:
                self.state.ability_cooldowns[key] = time.time()

            # Add delay if duration specified
            if duration:
                time.sleep(duration * 0.1)  # Scale down for tests

            return success

        except Exception as e:
            self.logger.error(f"Error recording key press: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def get_screen_region(self):
        """Get current game state data"""
        try:
            state_data = {
                'health': self.state.health,
                'current_position': self.state.current_position,
                'in_combat': self.state.is_in_combat,
                'resources': self.state.detected_resources,
                'obstacles': self.state.detected_obstacles,
                'fish_bite_active': self.state.fish_bite_active,
                'screen_content': self.state.screen_content,
                'is_mounted': self.state.is_mounted,
                'ability_cooldowns': self.state.ability_cooldowns,
                'terrain_type': self.state.terrain_type
            }
            return state_data

        except Exception as e:
            self.logger.error(f"Error getting screen region: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def set_game_state(self, **kwargs):
        """Update game state with terrain handling"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    if key == 'terrain_type':
                        # Update speed multiplier when terrain changes
                        self.state.terrain_speed_multiplier = self.terrain_speeds.get(value, 1.0)
                    setattr(self.state, key, value)

            self.logger.info(f"Updated game state: {kwargs}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating game state: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def start_simulation(self):
        """Start background simulation"""
        self.running = True
        self.simulation_thread = Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True  # Allow clean shutdown
        self.simulation_thread.start()
        self.logger.info("Mock environment simulation started")

    def stop_simulation(self):
        """Stop background simulation"""
        self.running = False
        if hasattr(self, 'simulation_thread'):
            self.simulation_thread.join(timeout=1.0)  # Wait with timeout
        self.logger.info("Mock environment simulation stopped")

    def _run_simulation(self):
        """Simulate game events"""
        bite_cooldown = 0
        while self.running:
            current_time = time.time()

            # Handle fish bite events
            if bite_cooldown <= 0 and random.random() < 0.1:  # 10% chance per cycle
                self.logger.debug("Triggering fish bite event")
                self.state.fish_bite_active = True
                self.fish_bite_event.set()

                # Short delay for bite detection
                time.sleep(0.2)

                self.state.fish_bite_active = False
                self.fish_bite_event.clear()
                bite_cooldown = 1.0  # 1 second cooldown
                self.logger.debug("Fish bite event completed")
            else:
                bite_cooldown = max(0, bite_cooldown - 0.1)

            # Simulate combat damage and other environmental effects
            if self.state.is_in_combat:
                self.state.health = max(0, self.state.health - random.uniform(1, 5))
                self.logger.debug(f"Combat damage taken, health now: {self.state.health}")

            time.sleep(0.05)  # Reduced CPU usage while maintaining responsiveness

    def heal(self, amount):
        """Heal character"""
        try:
            self.state.health = min(100, self.state.health + amount)
            self.logger.debug(f"Healed for {amount}, health now: {self.state.health}")
            return True

        except Exception as e:
            self.logger.error(f"Error healing: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return False

    def get_current_health(self):
        """Get current health value"""
        return self.state.health

    def is_mounted(self):
        """Check if character is mounted"""
        return self.state.is_mounted
    
    def get_mouse_pos(self):
        """Get current mouse position"""
        recent_moves = [e for e in self.input_events if e['type'] == 'mouse_move']
        if recent_moves:
            latest = recent_moves[-1]
            return (latest['x'], latest['y'])
        return (0, 0)


def create_test_environment():
    """Create and initialize mock environment"""
    env = MockEnvironment()
    env.min_action_interval = 0.0  # Disable timing restrictions for testing
    return env