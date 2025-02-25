"""Test script for Roboflow API integration"""
import logging
from inference_sdk import InferenceHTTPClient
import cv2
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('RoboflowTest')

def test_roboflow_api():
    try:
        # Initialize Roboflow client
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="dw3ovISGhOXZ5NzDkg7L"
        )
        logger.info("Roboflow client initialized")

        # Test image path
        image_path = 'attached_assets/image_1740444431862.png'
        if not os.path.exists(image_path):
            logger.error(f"Test image not found: {image_path}")
            return False

        # Load and prepare image
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to load image")
            return False

        # Convert to bytes
        is_success, img_buf = cv2.imencode('.jpg', img)
        if not is_success:
            logger.error("Failed to encode image")
            return False

        img_bytes = img_buf.tobytes()
        logger.debug(f"Image prepared, size: {len(img_bytes)} bytes")

        # Make API call
        logger.info("Making API call to Roboflow...")
        result = client.infer(img_bytes, model_id="albiongathering/3")

        logger.info(f"API Response: {result}")
        return True

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    success = test_roboflow_api()
    if success:
        logger.info("Roboflow API test completed successfully")
    else:
        logger.error("Roboflow API test failed")
        exit(1)