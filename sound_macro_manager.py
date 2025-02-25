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
        self.hotkeys_file = Path('macro_hotkeys.json')
        self.mappings: Dict[str, str] = {}
        self.hotkeys: Dict[str, str] = {}  # macro_name -> hotkey
        self._load_mappings()
        self._load_hotkeys()

        # Recording state
        self.is_recording = False
        self.last_recorded_sound = None
        self.current_recording_name = None

        self.logger.info("Sound macro manager initialized")

    def _load_hotkeys(self):
        """Load macro to hotkey mappings"""
        if self.hotkeys_file.exists():
            try:
                with open(self.hotkeys_file, 'r') as f:
                    self.hotkeys = json.load(f)
                self.logger.info(f"Loaded {len(self.hotkeys)} macro hotkey mappings")
            except Exception as e:
                self.logger.error(f"Error loading hotkey mappings: {e}")
                self.hotkeys = {}
        else:
            self.hotkeys = {}
            self._save_hotkeys()

    def _save_hotkeys(self):
        """Save macro to hotkey mappings"""
        try:
            with open(self.hotkeys_file, 'w') as f:
                json.dump(self.hotkeys, f, indent=2)
            self.logger.info(f"Saved {len(self.hotkeys)} macro hotkey mappings")
        except Exception as e:
            self.logger.error(f"Error saving hotkey mappings: {e}")

    def assign_hotkey(self, macro_name: str, hotkey: str) -> bool:
        """Assign a hotkey to trigger a specific macro"""
        try:
            # Verify macro file exists
            macro_file = self.macros_dir / f"{macro_name}.json"
            self.logger.info(f"Checking for macro file: {macro_file}")

            if not macro_file.exists():
                self.logger.error(f"Macro file '{macro_file}' not found")
                return False

            # Load existing macro data
            try:
                with open(macro_file, 'r') as f:
                    macro_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading macro file: {e}")
                return False

            # Remove this hotkey if it was assigned to another macro
            for existing_macro, existing_hotkey in self.hotkeys.items():
                if existing_hotkey == hotkey and existing_macro != macro_name:
                    self.logger.info(f"Removing hotkey '{hotkey}' from macro '{existing_macro}'")
                    del self.hotkeys[existing_macro]

            # Update macro file with hotkey
            macro_data['hotkey'] = hotkey
            try:
                with open(macro_file, 'w') as f:
                    json.dump(macro_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Error writing macro file: {e}")
                return False

            # Assign new hotkey
            self.hotkeys[macro_name] = hotkey
            self._save_hotkeys()
            self.logger.info(f"Successfully assigned hotkey '{hotkey}' to macro '{macro_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error assigning hotkey to macro: {e}")
            return False

    def remove_hotkey(self, macro_name: str) -> bool:
        """Remove a hotkey assignment from a macro"""
        if macro_name in self.hotkeys:
            del self.hotkeys[macro_name]
            self._save_hotkeys()
            self.logger.info(f"Removed hotkey for macro: {macro_name}")
            return True
        return False

    def get_macro_hotkey(self, macro_name: str) -> Optional[str]:
        """Get the hotkey assigned to a macro"""
        return self.hotkeys.get(macro_name)

    def get_all_hotkeys(self) -> Dict[str, str]:
        """Get all macro to hotkey mappings"""
        return self.hotkeys.copy()

    def trigger_macro_by_hotkey(self, hotkey: str) -> bool:
        """Execute a macro that's mapped to a specific hotkey"""
        try:
            # Find macro mapped to this hotkey
            for macro_name, mapped_hotkey in self.hotkeys.items():
                if mapped_hotkey == hotkey:
                    macro_file = self.macros_dir / f"{macro_name}.json"
                    if macro_file.exists():
                        self.logger.info(f"Executing macro '{macro_name}' triggered by hotkey '{hotkey}'")
                        return self.macro_tester.play_macro(str(macro_file))
                    else:
                        self.logger.error(f"Macro file not found: {macro_file}")
                        return False

            self.logger.info(f"No macro mapped to hotkey: {hotkey}")
            return False

        except Exception as e:
            self.logger.error(f"Error triggering macro by hotkey: {e}")
            return False

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

    def record_sound_trigger(self, name: str, duration: float = 2.0, save: bool = True) -> bool:
        """Record a new sound trigger"""
        if self.is_recording:
            self.logger.error("Already recording a sound")
            return False

        try:
            self.is_recording = True
            self.current_recording_name = name
            self.logger.info(f"Starting sound recording for: {name}")

            self.last_recorded_sound = self.sound_trigger.record_sound(name, duration)

            if save and self.last_recorded_sound is not None:
                return self.save_sound_trigger(name)

            return self.last_recorded_sound is not None
        except Exception as e:
            self.logger.error(f"Error recording sound trigger: {e}")
            return False
        finally:
            self.is_recording = False
            self.current_recording_name = None

    def stop_recording(self) -> bool:
        """Stop the current recording"""
        if not self.is_recording:
            self.logger.warning("No active recording to stop")
            return False

        try:
            if not self.test_mode:
                # Stop any active recording
                if self.sound_trigger:
                    self.sound_trigger.stop_recording()
            self.logger.info("Recording stopped successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            return False
        finally:
            self.is_recording = False
            self.current_recording_name = None

    def save_sound_trigger(self, name: str) -> bool:
        """Save the last recorded sound trigger"""
        try:
            if self.last_recorded_sound is None:
                self.logger.error("No sound data to save")
                return False

            return self.sound_trigger.save_trigger(name, self.last_recorded_sound)
        except Exception as e:
            self.logger.error(f"Error saving sound trigger: {e}")
            return False

    def record_macro(self, name: str, duration: float = 10.0) -> bool:
        """Record a new macro"""
        if self.is_recording:
            self.logger.error("Already recording")
            return False

        try:
            self.is_recording = True
            self.current_recording_name = name
            self.logger.info(f"Starting macro recording: {name}")

            if not self.macro_tester.start_recording():
                self.logger.error("Failed to start macro recording")
                return False

            time.sleep(duration)

            return self.macro_tester.stop_recording()
        except Exception as e:
            self.logger.error(f"Error recording macro: {e}")
            return False
        finally:
            self.is_recording = False
            self.current_recording_name = None

    def stop_macro_recording(self) -> bool:
        """Stop the current macro recording"""
        if not self.is_recording:
            self.logger.warning("No active macro recording to stop")
            return False

        try:
            success = self.macro_tester.stop_recording()
            if success:
                self.logger.info("Successfully stopped macro recording")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error stopping macro recording: {e}")
            return False
        finally:
            self.is_recording = False
            self.current_recording_name = None

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
        self.sound_trigger.start_monitoring(callback=self._handle_sound_detected)
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

    def _handle_sound_detected(self, sound_name: str):
        """Internal method to handle detected sounds"""
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