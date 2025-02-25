"""Test script for vision system"""
import logging
import cv2
import numpy as np
from vision_system import VisionSystem
import os
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('VisionTest')

def test_vision_system():
    """Test the vision system with game resource images"""
    # Initialize vision system
    vision = VisionSystem()

    # Define font once for reuse
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Test with sample images
    try:
        # Test specific resource images
        test_images = [
            ('attached_assets/image_1740461896066.png', 'Full Copper Ore Node'),
            ('attached_assets/image_1740461924307.png', 'Empty Copper Ore Node'),
            ('attached_assets/image_1740461943568.png', 'Full Limestone Node'),
            ('attached_assets/image_1740461978415.png', 'Empty Limestone Node')
        ]

        success = True
        for img_path, description in test_images:
            if not os.path.exists(img_path):
                logger.error(f"Test image not found: {img_path}")
                success = False
                continue

            logger.info(f"\nTesting detection on {description}: {img_path}")

            # Load test image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load test image: {img_path}")
                success = False
                continue

            # Test resource detection
            detections = vision.detect_resources(img)

            # Group detections by class and state
            class_groups = defaultdict(list)
            for det in detections:
                class_groups[det['class_id']].append(det)

            # Log detailed detection information
            logger.info(f"Total detections for {description}: {len(detections)}")
            for class_id, group in class_groups.items():
                avg_confidence = sum(d['confidence'] for d in group) / len(group)
                logger.info(f"  {class_id}: {len(group)} instances (avg conf: {avg_confidence:.2f})")
                for i, det in enumerate(group):
                    logger.debug(f"    Detection {i+1}: confidence={det['confidence']:.2f}, bbox={det['bbox']}")

            # Draw detections on image with enhanced visualization
            for det in detections:
                if det['bbox']:
                    x, y, w, h = map(int, det['bbox'])
                    # Draw bounding box with color based on confidence
                    confidence = det['confidence']
                    color = (
                        0,  # Blue
                        int(255 * confidence),  # Green (higher for better confidence)
                        int(255 * (1 - confidence))  # Red (lower for better confidence)
                    )
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                    # Add label with confidence and state
                    label = f"{det['class_id']} ({confidence:.2f})"
                    font_scale = 0.6
                    thickness = 2

                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness)

                    # Draw background rectangle for text
                    cv2.rectangle(img,
                                (x, y - text_height - 6),
                                (x + text_width + 2, y),
                                color, -1)

                    # Draw text in white
                    cv2.putText(img, label, (x, y - 5), font,
                              font_scale, (255, 255, 255), thickness)

            # Save result with clear naming
            output_path = f'detection_result_{Path(img_path).stem}.png'
            cv2.imwrite(output_path, img)
            logger.info(f"Saved detection visualization to: {output_path}")

        return success

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