"""Core implementation of the Fishing Bot"""
import time
import logging
import cv2
import numpy as np
import keyboard
from mss import mss
from pathlib import Path
from datetime import datetime
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fishing_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FishingBot')

class FishingBot:
    def __init__(self, test_mode=None):
        self.running = False
        # Auto-detect test mode if not specified
        if test_mode is None:
            test_mode = platform.system() != 'Windows'
        self.test_mode = test_mode

        if test_mode:
            logger.warning("Running in test mode (non-Windows platform)")
            # Mock screen capture in test mode
            self.screen = None
        else:
            try:
                self.screen = mss()
            except Exception as e:
                logger.error(f"Failed to initialize screen capture: {e}")
                raise

        self.logger = logging.getLogger('FishingBot')

        # Create screenshots directory
        self.screenshots_dir = Path('debug_screenshots')
        self.screenshots_dir.mkdir(exist_ok=True)

        # Load configuration
        self.monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
        self.CAST_KEY = 'f3'  # Default fishing key
        self.STOP_KEY = 'f6'  # Emergency stop key

        self.logger.info("Fishing bot initialized")

    def start(self):
        """Start the fishing bot"""
        self.running = True
        self.logger.info("Starting fishing bot...")

        try:
            keyboard.on_press_key(self.STOP_KEY, lambda _: self.stop())

            while self.running:
                self.fish_cycle()
                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error during fishing: {e}")
            self.stop()

    def stop(self):
        """Stop the fishing bot"""
        self.running = False
        self.logger.info("Stopping fishing bot...")

    def fish_cycle(self):
        """Execute one fishing cycle"""
        try:
            # Cast fishing line
            keyboard.press_and_release(self.CAST_KEY)
            self.logger.info("Cast fishing line")

            # Save debug screenshot before detection
            if not self.test_mode:
                self.save_debug_screenshot("before_detection")

            # Wait for bite detection
            time.sleep(2)  # Initial delay

            while self.running:
                if self.test_mode:
                    # Simulate bite detection in test mode
                    self.logger.info("Test mode: Simulating fish bite")
                    time.sleep(3)
                    return True

                screenshot = self.take_screenshot()
                if self.detect_bite(screenshot):
                    self.logger.info("Fish bite detected!")
                    self.save_debug_screenshot("bite_detected")
                    keyboard.press_and_release(self.CAST_KEY)
                    time.sleep(1)
                    break
                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in fishing cycle: {e}")

    def take_screenshot(self):
        """Capture screen for bite detection"""
        if self.test_mode:
            # Return mock screenshot in test mode
            return np.zeros((self.monitor["height"], self.monitor["width"], 3), dtype=np.uint8)

        screenshot = self.screen.grab(self.monitor)
        return np.array(screenshot)

    def detect_bite(self, screenshot):
        """Detect if there's a fish bite using image processing"""
        try:
            if self.test_mode:
                # Simulate detection in test mode
                return True

            # Convert to HSV color space
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

            # Define blue color range for the fishing indicator
            lower_blue = np.array([100, 150, 150])
            upper_blue = np.array([140, 255, 255])

            # Create mask for blue pixels
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Count blue pixels
            blue_pixels = np.sum(mask > 0)

            # Define threshold for bite detection
            BITE_THRESHOLD = 100

            if blue_pixels > BITE_THRESHOLD:
                self.logger.debug(f"Blue pixels detected: {blue_pixels}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in bite detection: {e}")
            return False

    def save_debug_screenshot(self, prefix):
        """Save a debug screenshot"""
        if self.test_mode:
            self.logger.debug(f"Test mode: Skipping screenshot save for {prefix}")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.screenshots_dir / f"{prefix}_{timestamp}.png"
            screenshot = self.take_screenshot()
            cv2.imwrite(str(filename), screenshot)
            self.logger.debug(f"Saved debug screenshot: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving debug screenshot: {e}")

def main():
    bot = FishingBot(test_mode=True)  # Force test mode for now
    logger.info("Press F3 to cast fishing line, F6 to stop")
    bot.start()

if __name__ == "__main__":
    main()