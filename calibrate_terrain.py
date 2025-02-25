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

        # Analyze deep water areas (darker blue)
        deep_water_mask = cv2.inRange(hsv, np.array([95, 150, 150]), np.array([125, 255, 255]))
        deep_water_img = cv2.bitwise_and(img, img, mask=deep_water_mask)
        cv2.imwrite('deep_water_detection.png', deep_water_img)

        # Analyze shallow water areas (lighter blue)
        shallow_water_mask = cv2.inRange(hsv, np.array([95, 60, 180]), np.array([125, 150, 255]))
        shallow_water_img = cv2.bitwise_and(img, img, mask=shallow_water_mask)
        cv2.imwrite('shallow_water_detection.png', shallow_water_img)

        # Analyze mountain areas (dark areas)
        mountain_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([30, 40, 160]))
        mountain_img = cv2.bitwise_and(img, img, mask=mountain_mask)
        cv2.imwrite('mountain_detection.png', mountain_img)

        # Analyze cliff areas (very dark areas)
        cliff_mask = cv2.inRange(hsv, np.array([0, 0, 20]), np.array([180, 25, 80]))
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