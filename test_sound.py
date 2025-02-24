"""Test script for sound recording and playback"""
import time
import logging
import platform
from sound_trigger import SoundTrigger

def test_record_and_play():
    """Test basic recording and playback functionality"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize sound trigger system with test mode
    test_mode = platform.system() != 'Windows'
    sound_trigger = SoundTrigger(test_mode=test_mode)

    try:
        # Record a test sound
        print("Recording a test sound in 3 seconds...")
        time.sleep(3)

        # Record for 2 seconds
        duration = 2.0
        if sound_trigger.record_sound("test_sound", duration=duration):
            print(f"Successfully recorded {duration} seconds of audio")
            time.sleep(1)  # Wait a moment before playback

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
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
    finally:
        # Cleanup only if we have a real audio object (not in test mode)
        if not test_mode and hasattr(sound_trigger, 'audio') and sound_trigger.audio:
            sound_trigger.audio.terminate()

if __name__ == "__main__":
    test_record_and_play()