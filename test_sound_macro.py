"""Test script for sound-triggered macro functionality"""
import time
import logging
import platform
from sound_macro_manager import SoundMacroManager

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize manager with appropriate modes
    test_mode = platform.system() != 'Windows'
    headless = platform.system() != 'Windows'
    manager = SoundMacroManager(test_mode=test_mode, headless=headless)

    try:
        # Record a test sound trigger
        print("Recording a test sound in 3 seconds...")
        time.sleep(3)
        if manager.record_sound_trigger("test_sound", duration=2.0):
            print(f"Successfully recorded sound trigger")

            # Record a test macro (simulated mouse movements)
            print("\nRecording a test macro in 3 seconds...")
            time.sleep(3)
            if manager.record_macro("test_macro", duration=5.0):
                print("Successfully recorded macro")

                # Assign macro to sound trigger
                if manager.assign_macro_to_sound("test_sound", "test_macro"):
                    print("\nSuccessfully mapped sound to macro")

                    # Start monitoring
                    print("\nStarting sound monitoring (will run for 30 seconds)...")
                    manager.start_monitoring()

                    # In test mode, this will trigger random detections
                    time.sleep(30)

                    # Stop monitoring
                    manager.stop_monitoring()
                    print("\nStopped monitoring")
                else:
                    print("Failed to map sound to macro")
            else:
                print("Failed to record macro")
        else:
            print("Failed to record sound")

    except KeyboardInterrupt:
        print("\nStopped by user")
        manager.stop_monitoring()

    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        manager.stop_monitoring()

if __name__ == "__main__":
    main()
