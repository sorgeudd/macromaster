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
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger('VisionTest')

def test_vision_system():
    """Test the vision system with sample images"""
    # Initialize vision system
    vision = VisionSystem()

    # Test with sample images
    try:
        # Test specific image for titanium
        img_path = 'attached_assets/image_1740444431862.png'

        if not os.path.exists(img_path):
            logger.error(f"Test image not found: {img_path}")
            return False

        logger.info(f"\nTesting titanium detection with image: {img_path}")

        # Load test image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load test image: {img_path}")
            return False

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
            # Log individual detection details
            for i, det in enumerate(group):
                logger.info(f"    Detection {i+1}: confidence={det['confidence']:.2f}, bbox={det['bbox']}")

        # Draw detections on image with enhanced visualization
        for det in detections:
            if det['bbox']:
                x, y, w, h = map(int, det['bbox'])
                # Draw bounding box in red for better visibility
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Add label with confidence
                label = f"{det['class_id']} {det['confidence']:.2f}"
                font_scale = 0.6
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness)

                # Draw background rectangle for text
                cv2.rectangle(img, 
                            (x, y - text_height - 6), 
                            (x + text_width + 2, y),
                            (0, 0, 255), -1)

                # Draw text in white
                cv2.putText(img, label, (x, y - 5), font,
                          font_scale, (255, 255, 255), thickness)

        # Save result with clear naming
        output_path = 'titanium_detection_result.png'
        cv2.imwrite(output_path, img)
        logger.info(f"Saved detection visualization to: {output_path}")

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