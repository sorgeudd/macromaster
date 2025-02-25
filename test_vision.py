"""Test script for vision system"""
import cv2
from PIL import Image
import numpy as np
from vision_system import VisionSystem
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VisionTest')

def test_vision_system():
    """Test the vision system with sample images"""
    # Initialize vision system
    vision = VisionSystem()

    # Test with sample images
    try:
        # Test paths
        image_paths = [
            'attached_assets/image_1740444431862.png',
            'attached_assets/image_1740444463467.png'
        ]

        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.error(f"Test image not found: {img_path}")
                continue

            logger.info(f"Testing with image: {img_path}")

            # Load test image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load test image: {img_path}")
                continue

            # Test resource detection
            detections = vision.detect_resources(img)
            logger.info(f"Detected {len(detections)} objects")

            # Draw detections on image
            for det in detections:
                if det['bbox']:
                    x, y, w, h = map(int, det['bbox'])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{det['class_id']} {det['confidence']:.2f}"
                    cv2.putText(img, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save result
            output_path = f'detection_result_{Path(img_path).stem}.png'
            cv2.imwrite(output_path, img)
            logger.info(f"Saved detection result to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_vision_system()
    if success:
        logger.info("Vision system test completed successfully")
    else:
        logger.error("Vision system test failed")
        exit(1)