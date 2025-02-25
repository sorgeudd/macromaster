"""Test script for sound-triggered macro functionality"""
import time
import logging
import platform
from sound_macro_manager import SoundMacroManager

def test_sound_macro(recording_source="mic"):
    """Test sound macro functionality with specified recording source"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize manager with appropriate modes
    test_mode = platform.system() != 'Windows'
    headless = platform.system() != 'Windows'
    manager = SoundMacroManager(test_mode=test_mode, headless=headless, recording_source=recording_source)

    try:
        # Record a test sound trigger
        print(f"\nTesting {recording_source} recording...")
        print("Recording a test sound in 3 seconds...")
        time.sleep(3)
        if manager.record_sound_trigger(f"test_sound_{recording_source}", duration=2.0):
            print("Successfully recorded sound trigger")

            # Record a test macro
            print("\nRecording a test macro in 3 seconds...")
            time.sleep(3)
            if manager.record_macro(f"test_macro_{recording_source}", duration=5.0):
                print("Successfully recorded macro")

                # Assign macro to sound trigger
                if manager.assign_macro_to_sound(f"test_sound_{recording_source}", f"test_macro_{recording_source}"):
                    print("\nSuccessfully mapped sound to macro")

                    # Test sound trigger callback directly
                    print("\nTesting sound trigger callback directly...")
                    manager._handle_sound_detected(f"test_sound_{recording_source}")
                    time.sleep(2)  # Wait for macro playback

                    # Start monitoring
                    print("\nStarting sound monitoring for 10 seconds...")
                    if manager.start_monitoring():
                        time.sleep(10)  # Monitor for 10 seconds
                        manager.stop_monitoring()
                        print("\nStopped monitoring")
                        return True
                    else:
                        print("Failed to start monitoring")
                else:
                    print("Failed to map sound to macro")
            else:
                print("Failed to record macro")
        else:
            print("Failed to record sound")

        return False

    except KeyboardInterrupt:
        print("\nStopped by user")
        manager.stop_monitoring()
        return False

    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        manager.stop_monitoring()
        return False

def main():
    # Test both mic and system audio recording
    sources = ["mic", "system"]
    results = {}

    for source in sources:
        print(f"\n=== Testing {source.upper()} recording ===")
        success = test_sound_macro(source)
        results[source] = "Success" if success else "Failed"
        time.sleep(1)  # Brief pause between tests

    # Print summary
    print("\n=== Test Results ===")
    for source, result in results.items():
        print(f"{source.upper()}: {result}")

    # Return overall success
    return all(result == "Success" for result in results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)