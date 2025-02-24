"""Sound-triggered macro management module"""
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Callable
from test_macro import MacroTester
from sound_trigger import SoundTrigger

class SoundMacroManager:
    def __init__(self, test_mode: bool = False, headless: bool = False):
        self.logger = logging.getLogger('SoundMacroManager')
        self.test_mode = test_mode
        self.headless = headless

        # Initialize components
        self.macro_tester = MacroTester(test_mode=test_mode, headless=headless)
        self.sound_trigger = SoundTrigger()

        # Load existing mappings
        self.mappings_file = Path('sound_macro_mappings.json')
        self.mappings: Dict[str, str] = {}
        self._load_mappings()

        self.logger.info("Sound macro manager initialized")

    def _load_mappings(self):
        """Load sound trigger to macro mappings"""
        if self.mappings_file.exists():
            try:
                with open(self.mappings_file, 'r') as f:
                    self.mappings = json.load(f)
                self.logger.info(f"Loaded {len(self.mappings)} sound-macro mappings")
            except Exception as e:
                self.logger.error(f"Error loading mappings: {e}")
                self.mappings = {}

    def _save_mappings(self):
        """Save sound trigger to macro mappings"""
        try:
            with open(self.mappings_file, 'w') as f:
                json.dump(self.mappings, f, indent=2)
            self.logger.info(f"Saved {len(self.mappings)} sound-macro mappings")
        except Exception as e:
            self.logger.error(f"Error saving mappings: {e}")

    def record_sound_trigger(self, name: str, duration: float = 2.0) -> bool:
        """Record a new sound trigger"""
        return self.sound_trigger.record_sound(name, duration)

    def play_sound_trigger(self, name: str) -> bool:
        """Play back a recorded sound trigger"""
        return self.sound_trigger.play_sound(name)

    def record_macro(self, name: str, duration: float = 10.0) -> bool:
        """Record a new macro"""
        if not self.macro_tester.start_recording():
            return False

        import time
        time.sleep(duration)

        return self.macro_tester.stop_recording()

    def assign_macro_to_sound(self, sound_name: str, macro_name: str) -> bool:
        """Assign a macro to be triggered by a specific sound"""
        if sound_name not in self.sound_trigger.get_trigger_names():
            self.logger.error(f"Sound trigger '{sound_name}' not found")
            return False

        try:
            # Verify macro file exists
            macro_file = Path(f"{macro_name}.json")
            if not macro_file.exists():
                self.logger.error(f"Macro file '{macro_file}' not found")
                return False

            # Add mapping
            self.mappings[sound_name] = macro_name
            self._save_mappings()
            return True

        except Exception as e:
            self.logger.error(f"Error assigning macro to sound: {e}")
            return False

    def start_monitoring(self) -> bool:
        """Start monitoring for sound triggers and execute mapped macros"""
        def on_sound_detected(sound_name: str):
            if sound_name in self.mappings:
                macro_name = self.mappings[sound_name]
                self.logger.info(f"Executing macro '{macro_name}' triggered by sound '{sound_name}'")
                self.macro_tester.play_macro(f"{macro_name}.json")

        self.sound_trigger.start_monitoring(callback=on_sound_detected)
        return True

    def stop_monitoring(self):
        """Stop monitoring for sound triggers"""
        self.sound_trigger.stop_monitoring()

    def list_mappings(self) -> Dict[str, str]:
        """Get current sound trigger to macro mappings"""
        return self.mappings.copy()

    def remove_mapping(self, sound_name: str) -> bool:
        """Remove a sound trigger to macro mapping"""
        if sound_name in self.mappings:
            del self.mappings[sound_name]
            self._save_mappings()
            return True
        return False
