"""Sound recording and detection module for macro triggers

This module provides functionality for recording and detecting sound triggers,
supporting both microphone input and system audio output recording.

Key Features:
- Record sound triggers from microphone or system audio
- Play back recorded sounds
- Monitor for sound patterns and trigger callbacks
- Save and load sound triggers
- Cross-platform support with test mode
"""
import os
import time
import json
import logging
import threading
import random
from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np
import pyaudio

class SoundTrigger:
    def __init__(self, sample_rate: int = 44100, threshold: float = 0.75, test_mode: bool = False, recording_source: str = "mic"):
        """Initialize sound trigger system

        Args:
            sample_rate: Audio sample rate (Hz)
            threshold: Detection threshold for triggers (0.0-1.0)
            test_mode: Run in test mode without real audio
            recording_source: Audio source ("mic" or "system")
        """
        self.logger = logging.getLogger('SoundTrigger')
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.recording = False
        self.monitoring = False
        self.playing = False
        self.test_mode = test_mode
        self.recording_source = recording_source.lower()

        if self.recording_source not in ["mic", "system"]:
            self.logger.warning(f"Invalid recording source '{recording_source}', defaulting to 'mic'")
            self.recording_source = "mic"

        # Audio settings
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1

        # Initialize PyAudio
        try:
            if not self.test_mode:
                self.audio = pyaudio.PyAudio()
                try:
                    # Try to get default input device
                    self.audio.get_default_input_device_info()
                    # Find system playback device for loopback
                    self.system_device = self._find_system_device()
                    if self.recording_source == "system" and self.system_device is None:
                        self.logger.warning("No system audio device found, falling back to microphone")
                        self.recording_source = "mic"
                except (IOError, OSError):
                    self.logger.warning(f"No input device found for {self.recording_source}, forcing test mode")
                    self.test_mode = True
                    self.audio = None
            else:
                self.logger.info("Running in test mode")
                self.audio = None
        except Exception as e:
            self.logger.warning(f"Audio hardware not available, forcing test mode: {e}")
            self.test_mode = True
            self.audio = None

        # Create sounds directory if it doesn't exist
        self.sounds_dir = Path('sounds')
        self.sounds_dir.mkdir(exist_ok=True)

        # Load existing sound triggers
        self.triggers: Dict[str, Dict] = {}
        self._load_triggers()

        self.logger.info(f"Sound trigger system initialized (test mode: {self.test_mode}, source: {self.recording_source})")

    def _find_system_device(self):
        """Find system playback device for loopback recording"""
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                # Look for system playback device (usually "Speakers" or similar)
                if device_info.get('maxInputChannels') == 0 and device_info.get('maxOutputChannels') > 0:
                    self.logger.debug(f"Found system playback device: {device_info.get('name')}")
                    return i
            return None
        except Exception as e:
            self.logger.error(f"Error finding system device: {e}")
            return None

    def _load_triggers(self):
        """Load saved trigger configurations"""
        trigger_file = self.sounds_dir / 'triggers.json'
        if trigger_file.exists():
            try:
                with open(trigger_file, 'r') as f:
                    self.triggers = json.load(f)
                self.logger.info(f"Loaded {len(self.triggers)} sound triggers")
            except Exception as e:
                self.logger.error(f"Error loading triggers: {e}")
                self.triggers = {}

    def _save_triggers(self):
        """Save trigger configurations"""
        trigger_file = self.sounds_dir / 'triggers.json'
        try:
            with open(trigger_file, 'w') as f:
                json.dump(self.triggers, f, indent=2)
            self.logger.info(f"Saved {len(self.triggers)} sound triggers")
        except Exception as e:
            self.logger.error(f"Error saving triggers: {e}")

    def record_sound(self, name: str, duration: float = 2.0) -> bool:
        """Record a new sound trigger
        Args:
            name: Name of the sound trigger
            duration: Recording duration in seconds
        """
        if self.recording:
            self.logger.warning("Already recording")
            return False

        try:
            self.recording = True

            if self.test_mode:
                # Generate a test signal in test mode
                time_points = np.linspace(0, duration, int(self.sample_rate * duration))
                test_signal = np.sin(2 * np.pi * 440 * time_points)  # 440 Hz sine wave
                audio_data = test_signal.astype(np.float32)
                self.logger.info(f"Test mode: Generated {duration}s test signal")
            else:
                frames = []
                stream_kwargs = {
                    'format': self.format,
                    'channels': self.channels,
                    'rate': self.sample_rate,
                    'input': True,
                    'frames_per_buffer': self.chunk_size
                }

                # Configure for system audio if selected
                if self.recording_source == "system" and self.system_device is not None:
                    stream_kwargs.update({
                        'input_device_index': self.system_device,
                        'as_loopback': True  # Enable WASAPI loopback
                    })
                    self.logger.info("Recording system audio output")
                else:
                    self.logger.info("Recording from microphone")

                # Open recording stream
                stream = self.audio.open(**stream_kwargs)

                self.logger.info(f"Recording sound '{name}' for {duration} seconds...")

                # Record audio
                for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))

                # Stop recording
                stream.stop_stream()
                stream.close()
                audio_data = np.concatenate(frames)

            # Save audio file
            filename = self.sounds_dir / f"{name}.npy"
            np.save(filename, audio_data)

            # Extract features and save trigger
            features = self._extract_features(audio_data)
            self.triggers[name] = {
                'file': str(filename),
                'features': features.tolist(),
                'duration': duration,
                'source': self.recording_source
            }
            self._save_triggers()

            self.logger.info(f"Saved sound trigger '{name}' from {self.recording_source}")
            self.recording = False
            return True

        except Exception as e:
            self.logger.error(f"Error recording sound: {e}")
            self.recording = False
            return False

    def play_sound(self, name: str) -> bool:
        """Play back a recorded sound"""
        if self.playing:
            self.logger.warning("Already playing audio")
            return False

        if name not in self.triggers:
            self.logger.error(f"Sound '{name}' not found")
            return False

        try:
            self.playing = True
            filename = self.triggers[name]['file']

            # Load audio file
            audio_data = np.load(filename)

            if self.test_mode:
                # In test mode, just simulate playback
                self.logger.info(f"Test mode: Simulating playback of '{name}'")
                time.sleep(self.triggers[name]['duration'])
            else:
                # Open playback stream
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )

                # Play audio in chunks
                chunks = np.array_split(audio_data, len(audio_data) // self.chunk_size + 1)
                for chunk in chunks:
                    if not self.playing:
                        break
                    stream.write(chunk.astype(np.float32).tobytes())

                # Clean up
                stream.stop_stream()
                stream.close()

            self.playing = False
            self.logger.info(f"Finished playing sound '{name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error playing sound: {e}")
            self.playing = False
            return False

    def stop_playback(self):
        """Stop current playback"""
        self.playing = False

    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract audio features using FFT"""
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Calculate FFT
        fft = np.fft.rfft(audio_data)

        # Get magnitude spectrum
        magnitude = np.abs(fft)

        # Reduce dimensionality by averaging frequency bands
        n_bands = 13
        band_size = len(magnitude) // n_bands
        features = np.array([np.mean(magnitude[i:i+band_size])
                           for i in range(0, len(magnitude), band_size)])[:n_bands]

        # Normalize features
        features = features / np.max(features)
        return features

    def _compare_sounds(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two sets of audio features"""
        # Compute cosine similarity
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return float(similarity)

    def start_monitoring(self, callback: Callable[[str], None]):
        """Start monitoring for sound triggers"""
        if self.monitoring:
            self.logger.warning("Already monitoring")
            return

        def monitor_thread():
            if self.test_mode:
                while self.monitoring:
                    time.sleep(1)  # Check every second in test mode
                    # Simulate sound detection for testing
                    for name in self.triggers:
                        if random.random() < 0.1:  # 10% chance of detection
                            self.logger.info(f"Test mode: Simulated detection of sound '{name}'")
                            callback(name)
            else:
                stream_kwargs = {
                    'format': self.format,
                    'channels': self.channels,
                    'rate': self.sample_rate,
                    'input': True,
                    'frames_per_buffer': self.chunk_size
                }

                # Configure for system audio if selected
                if self.recording_source == "system" and self.system_device is not None:
                    stream_kwargs.update({
                        'input_device_index': self.system_device,
                        'as_loopback': True
                    })
                    self.logger.info("Monitoring system audio output")
                else:
                    self.logger.info("Monitoring microphone input")

                stream = self.audio.open(**stream_kwargs)

                while self.monitoring:
                    frames = []
                    # Record a short sample
                    for _ in range(0, int(self.sample_rate / self.chunk_size * 2.0)):  # 2-second window
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        frames.append(np.frombuffer(data, dtype=np.float32))

                    # Check for matches
                    audio_data = np.concatenate(frames)
                    features = self._extract_features(audio_data)

                    # Compare with all triggers
                    for name, trigger in self.triggers.items():
                        similarity = self._compare_sounds(features, np.array(trigger['features']))
                        if similarity > self.threshold:
                            self.logger.info(f"Detected sound trigger: {name} (similarity: {similarity:.2f})")
                            callback(name)

                stream.stop_stream()
                stream.close()

        self.monitoring = True
        threading.Thread(target=monitor_thread, daemon=True).start()
        self.logger.info(f"Started sound monitoring ({self.recording_source})")

    def stop_monitoring(self):
        """Stop monitoring for sound triggers"""
        self.monitoring = False
        self.logger.info("Stopped sound monitoring")

    def get_trigger_names(self) -> list:
        """Get list of available trigger names"""
        return list(self.triggers.keys())

    def remove_trigger(self, name: str) -> bool:
        """Remove a sound trigger"""
        if name not in self.triggers:
            return False

        try:
            # Remove audio file
            filepath = Path(self.triggers[name]['file'])
            if filepath.exists():
                filepath.unlink()

            # Remove from triggers
            del self.triggers[name]
            self._save_triggers()

            self.logger.info(f"Removed sound trigger: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Error removing trigger: {e}")
            return False

    def __del__(self):
        """Cleanup PyAudio"""
        try:
            self.monitoring = False
            self.playing = False
            if self.audio:
                self.audio.terminate()
        except Exception:
            pass