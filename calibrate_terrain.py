"""Script to calibrate terrain detection color ranges"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TerrainCalibration')

def analyze_map_colors(map_path):
    """Analyze color ranges in map for terrain features"""
    try:
        # Load map image
        img = cv2.imread(str(map_path))
        if img is None:
            logger.error("Failed to load map image")
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Analyze water areas (blue tones)
        water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
        water_img = cv2.bitwise_and(img, img, mask=water_mask)
        cv2.imwrite('water_detection.png', water_img)

        # Analyze mountain areas (dark areas)
        mountain_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([30, 50, 180]))
        mountain_img = cv2.bitwise_and(img, img, mask=mountain_mask)
        cv2.imwrite('mountain_detection.png', mountain_img)

        # Analyze cliff areas (very dark areas)
        cliff_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 30, 100]))
        cliff_img = cv2.bitwise_and(img, img, mask=cliff_mask)
        cv2.imwrite('cliff_detection.png', cliff_img)

        logger.info("Created terrain detection visualizations")
        return True

    except Exception as e:
        logger.error(f"Error analyzing map colors: {e}")
        return False

if __name__ == "__main__":
    map_path = Path("maps/default_map.png")
    if not map_path.exists():
        map_path = Path("attached_assets/image_1740466625160.png")
    
    if analyze_map_colors(map_path):
        logger.info("Terrain color analysis complete")
    else:
        logger.error("Failed to analyze terrain colors")
