"""Sound recording and detection module for macro triggers"""
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
import platform

class SoundTrigger:
    def __init__(self, sample_rate: int = 44100, threshold: float = 0.75):
        self.logger = logging.getLogger('SoundTrigger')
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.recording = False
        self.monitoring = False
        self.playing = False

        # Audio settings
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1

        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            self.test_mode = False
        except (ImportError, OSError) as e:
            self.logger.warning(f"Audio hardware not available, running in test mode: {e}")
            self.test_mode = True
            self.audio = None

        # Create sounds directory if it doesn't exist
        self.sounds_dir = Path('sounds')
        self.sounds_dir.mkdir(exist_ok=True)

        # Load existing sound triggers
        self.triggers: Dict[str, Dict] = {}
        self._load_triggers()

        self.logger.info(f"Sound trigger system initialized (test mode: {self.test_mode})")

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
        """Record a new sound trigger"""
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
            else:
                frames = []
                # Open recording stream
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )

                self.logger.info(f"Recording sound '{name}' for {duration} seconds...")

                # Record audio
                for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                    data = stream.read(self.chunk_size)
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
                'duration': duration
            }
            self._save_triggers()

            self.logger.info(f"Saved sound trigger '{name}'")
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
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )

                while self.monitoring:
                    frames = []
                    # Record a short sample
                    for _ in range(0, int(self.sample_rate / self.chunk_size * 2.0)):  # 2-second window
                        data = stream.read(self.chunk_size)
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
        self.logger.info("Started sound monitoring")

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