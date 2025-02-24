"""Sound-triggered macro management module"""
import logging
import json
import time
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
        self.sound_trigger = SoundTrigger(test_mode=test_mode)

        # Create macros directory if it doesn't exist
        self.macros_dir = Path('macros')
        self.macros_dir.mkdir(exist_ok=True)
        self.logger.info(f"Using macros directory: {self.macros_dir}")

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

    def record_macro(self, name: str, duration: float = 10.0) -> bool:
        """Record a new macro"""
        self.logger.info(f"Starting macro recording: {name}")
        if not self.macro_tester.start_recording():
            self.logger.error("Failed to start macro recording")
            return False

        time.sleep(duration)

        if self.macro_tester.stop_recording():
            # Move the recorded macro to our macros directory
            source = Path('recorded_macro.json')
            target = self.macros_dir / f"{name}.json"

            self.logger.info(f"Looking for recorded macro at: {source}")
            if source.exists():
                try:
                    # Read the source file first
                    with open(source, 'r') as f:
                        macro_data = json.load(f)

                    # Write to the target file
                    with open(target, 'w') as f:
                        json.dump(macro_data, f, indent=2)

                    # Remove the source file
                    source.unlink()

                    self.logger.info(f"Successfully saved macro as {target}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error saving macro file: {e}")
                    return False
            self.logger.error(f"Recorded macro file not found at {source}")
            return False

        self.logger.error("Failed to stop macro recording")
        return False

    def assign_macro_to_sound(self, sound_name: str, macro_name: str) -> bool:
        """Assign a macro to be triggered by a specific sound"""
        if sound_name not in self.sound_trigger.get_trigger_names():
            self.logger.error(f"Sound trigger '{sound_name}' not found")
            return False

        try:
            # Verify macro file exists
            macro_file = self.macros_dir / f"{macro_name}.json"
            self.logger.info(f"Checking for macro file: {macro_file}")

            if not macro_file.exists():
                self.logger.error(f"Macro file '{macro_file}' not found")
                return False

            # Add mapping
            self.mappings[sound_name] = macro_name
            self._save_mappings()
            self.logger.info(f"Successfully mapped sound '{sound_name}' to macro '{macro_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error assigning macro to sound: {e}")
            return False

    def start_monitoring(self) -> bool:
        """Start monitoring for sound triggers and execute mapped macros"""
        def on_sound_detected(sound_name: str):
            self.logger.info(f"Sound trigger detected: {sound_name}")

            if sound_name in self.mappings:
                macro_name = self.mappings[sound_name]
                macro_file = self.macros_dir / f"{macro_name}.json"

                self.logger.info(f"Found mapping for sound '{sound_name}' -> macro '{macro_name}'")
                self.logger.info(f"Looking for macro file: {macro_file}")

                if macro_file.exists():
                    self.logger.info(f"Executing macro '{macro_name}' triggered by sound '{sound_name}'")
                    self.macro_tester.play_macro(str(macro_file))
                else:
                    self.logger.error(f"Mapped macro file not found: {macro_file}")
            else:
                self.logger.info(f"No macro mapping found for sound: {sound_name}")

        self.sound_trigger.start_monitoring(callback=on_sound_detected)
        self.logger.info("Started sound trigger monitoring")
        return True

    def stop_monitoring(self):
        """Stop monitoring for sound triggers"""
        self.sound_trigger.stop_monitoring()
        self.logger.info("Stopped sound trigger monitoring")

    def list_mappings(self) -> Dict[str, str]:
        """Get current sound trigger to macro mappings"""
        return self.mappings.copy()

    def remove_mapping(self, sound_name: str) -> bool:
        """Remove a sound trigger to macro mapping"""
        if sound_name in self.mappings:
            del self.mappings[sound_name]
            self._save_mappings()
            self.logger.info(f"Removed mapping for sound: {sound_name}")
            return True
        return False