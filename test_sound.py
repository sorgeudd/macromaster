"""Test script for sound recording and playback"""
import time
import numpy as np
from sound_trigger import SoundTrigger

def test_record_and_play():
    """Test basic recording and playback functionality"""
    # Initialize sound trigger system
    sound_trigger = SoundTrigger()

    try:
        # Record a test sound
        print("Recording a test sound in 3 seconds...")
        time.sleep(3)

        # Record for 2 seconds
        duration = 2.0
        if sound_trigger.record_sound("test_sound", duration=duration):
            print(f"Successfully recorded {duration} seconds of audio")

            # Play back the recorded sound
            print("Playing back the recorded sound...")
            if sound_trigger.play_sound("test_sound"):
                print("Playback completed successfully")
            else:
                print("Playback failed")
        else:
            print("Recording failed")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Cleanup
        if hasattr(sound_trigger, 'audio'):
            sound_trigger.audio.terminate()

if __name__ == "__main__":
    test_record_and_play()