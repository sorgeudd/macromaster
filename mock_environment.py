"""Mock environment for headless testing of fishing bot functionality"""
import logging
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
    last_action_time: float = None  # Track timing of last action
    ability_cooldowns: dict = None  # Track ability cooldowns

    def __post_init__(self):
        if self.detected_resources is None:
            self.detected_resources = []
        if self.detected_obstacles is None:
            self.detected_obstacles = []
        if self.screen_content is None:
            # Create a mock screen content (black screen)
            self.screen_content = np.zeros((600, 800, 3), dtype=np.uint8)
        if self.ability_cooldowns is None:
            self.ability_cooldowns = {
                'e': 0.0,  # Last use time for each ability
                'w': 0.0,
                'q': 0.0,
                'space': 0.0
            }
        self.last_action_time = time.time()
        self.combat_start_time = time.time()

class MockEnvironment:
    def __init__(self):
        self.logger = logging.getLogger('MockEnvironment')
        self.state = GameState()
        self.input_events = []
        self.fish_bite_event = Event()
        self.running = False
        self.min_action_interval = 0.05  # Reduced for more responsive testing

        # Initialize mock screen
        self.window_size = (800, 600)
        self.state.screen_content = np.zeros((*self.window_size, 3), dtype=np.uint8)

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

            # Only generate events if enough time has passed since last action
            if current_time - self.state.last_action_time >= self.min_action_interval:
                # Simulate random fish bites with cooldown
                if bite_cooldown <= 0 and random.random() < 0.3:  # 30% chance per cycle
                    self.state.fish_bite_active = True
                    self.fish_bite_event.set()
                    time.sleep(0.1)  # Reduced for testing
                    self.state.fish_bite_active = False
                    self.fish_bite_event.clear()
                    bite_cooldown = 0.5  # Reduced for testing
                else:
                    bite_cooldown = max(0, bite_cooldown - 0.1)

                time.sleep(0.05)  # Reduced CPU usage while maintaining responsiveness

    def record_input(self, input_type, **kwargs):
        """Record input events for verification"""
        current_time = time.time()
        if current_time - self.state.last_action_time >= self.min_action_interval:
            event = {'type': input_type, 'timestamp': current_time, **kwargs}
            self.input_events.append(event)
            self.logger.debug(f"Recorded input event: {event}")
            self.state.last_action_time = current_time
            return True
        self.logger.warning(f"Action ignored - too soon after last action (interval: {self.min_action_interval}s)")
        return False

    def move_mouse(self, x, y):
        """Record mouse movement"""
        current_time = time.time()
        event = {
            'type': 'mouse_move',
            'timestamp': current_time,
            'x': x,
            'y': y
        }
        self.input_events.append(event)
        self.logger.debug(f"Recorded mouse move to ({x}, {y}) (total events: {len(self.input_events)})")
        self.state.last_action_time = current_time
        return True

    def click(self, button='left', clicks=1):
        """Record mouse click with improved logging"""
        try:
            current_time = time.time()
            event = {
                'type': 'mouse_click',
                'timestamp': current_time,
                'button': button,
                'clicks': clicks
            }
            self.input_events.append(event)
            self.logger.debug(f"Recorded mouse click: {button} button with {clicks} clicks")
            self.state.last_action_time = current_time
            return True

        except Exception as e:
            self.logger.error(f"Error recording click: {str(e)}")
            return False

    def press_key(self, key, duration=None):
        """Record key press with improved logging"""
        try:
            current_time = time.time()
            event = {'type': 'key_press', 'timestamp': current_time, 
                    'key': key, 'duration': duration}
            self.input_events.append(event)

            # Handle special keys
            if key == 'a':
                self.state.is_mounted = not self.state.is_mounted
                self.logger.debug(f"Mount state toggled: {'mounted' if self.state.is_mounted else 'dismounted'}")
            elif key == 'r' and self.state.fish_bite_active:
                self.logger.debug("Reeling triggered on fish bite")

            # Update ability cooldowns
            if key in self.state.ability_cooldowns:
                self.state.ability_cooldowns[key] = current_time
                self.logger.debug(f"Updated cooldown for ability '{key}'")

            self.state.last_action_time = current_time
            return True

        except Exception as e:
            self.logger.error(f"Error recording key press: {str(e)}")
            return False

    def get_screen_region(self):
        """Get current game state data"""
        return {
            'health': self.state.health,
            'current_position': self.state.current_position,
            'in_combat': self.state.is_in_combat,
            'resources': self.state.detected_resources,
            'obstacles': self.state.detected_obstacles,
            'fish_bite_active': self.state.fish_bite_active,
            'screen_content': self.state.screen_content,
            'is_mounted': self.state.is_mounted,
            'ability_cooldowns': self.state.ability_cooldowns
        }

    def set_game_state(self, **kwargs):
        """Update game state for testing"""
        current_time = time.time()
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                if key == 'is_in_combat' and value:
                    self.state.combat_start_time = current_time
                elif key == 'fish_bite_active':
                    if value:
                        self.fish_bite_event.set()
                    else:
                        self.fish_bite_event.clear()
        self.logger.info(f"Updated game state: {kwargs}")
        self.state.last_action_time = current_time

def create_test_environment():
    """Create and return a mock environment instance"""
    env = MockEnvironment()
    env.min_action_interval = 0.0  # Set minimum interval to 0 for testing
    return env