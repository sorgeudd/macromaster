"""Test script for vision system"""
import cv2
from PIL import Image
import numpy as np
from vision_system import VisionSystem
import logging
import os
from pathlib import Path
from collections import defaultdict

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

            logger.info(f"\nTesting with image: {img_path}")

            # Load test image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load test image: {img_path}")
                continue

            # Test resource detection
            detections = vision.detect_resources(img)

            # Group detections by class
            class_groups = defaultdict(list)
            for det in detections:
                class_groups[det['class_id']].append(det)

            # Log detailed detection information
            logger.info(f"Total detections: {len(detections)}")
            for class_id, group in class_groups.items():
                avg_confidence = sum(d['confidence'] for d in group) / len(group)
                logger.info(f"  {class_id}: {len(group)} instances (avg conf: {avg_confidence:.2f})")

            # Draw detections on image with enhanced visualization
            colors = {
                'wood': (0, 255, 0),    # Green
                'stone': (255, 0, 0),   # Blue
                'ore': (0, 0, 255),     # Red
                'fiber': (255, 255, 0),  # Cyan
                'hide': (128, 0, 128)   # Purple
            }

            for det in detections:
                if det['bbox']:
                    x, y, w, h = map(int, det['bbox'])
                    color = colors.get(det['class_id'], (0, 255, 0))

                    # Draw bounding box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                    # Add label with confidence
                    label = f"{det['class_id']} {det['confidence']:.2f}"
                    font_scale = 0.5
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness)

                    # Draw background rectangle for text
                    cv2.rectangle(img, 
                                (x, y - text_height - 6), 
                                (x + text_width, y),
                                color, -1)

                    # Draw text
                    cv2.putText(img, label, (x, y - 5), font,
                              font_scale, (255, 255, 255), thickness)

            # Save result
            output_path = f'detection_result_{Path(img_path).stem}.png'
            cv2.imwrite(output_path, img)
            logger.info(f"Saved detection result to: {output_path}")
            logger.info("-" * 50)

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