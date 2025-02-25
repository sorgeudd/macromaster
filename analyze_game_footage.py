"""Script to analyze the gameplay footage for arrow appearance and behavior"""
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('GameFootageAnalysis')

def analyze_video_frame(frame):
    """Analyze a single frame to detect arrow properties"""
    # Convert frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define initial color range for arrow detection
    lower_arrow = np.array([90, 160, 180])  # Current values
    upper_arrow = np.array([110, 200, 255])

    # Create mask and find contours
    mask = cv2.inRange(hsv, lower_arrow, upper_arrow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Analyze largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Get bounding rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)  # Changed from int0 to int32

        # Calculate dimensions
        width = rect[1][0]
        height = rect[1][1]
        angle = rect[2]

        logger.info(f"Arrow properties:")
        logger.info(f"Area: {area}")
        logger.info(f"Width x Height: {width} x {height}")
        logger.info(f"Angle: {angle}")

        # Draw detection results
        result = frame.copy()
        cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
        cv2.imwrite('arrow_detection.png', result)

        return area, (width, height), angle
    return None

def main():
    cap = cv2.VideoCapture('attached_assets/25.02.2025_10.10.43_REC.mp4')

    if not cap.isOpened():
        logger.error("Error: Could not open video file")
        return

    frame_count = 0
    arrow_properties = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            result = analyze_video_frame(frame)
            if result:
                arrow_properties.append(result)
                logger.info(f"Frame {frame_count}: {result}")

        frame_count += 1

        # Process first 100 frames only
        if frame_count >= 100:
            break

    cap.release()

    # Analyze collected data
    if arrow_properties:
        areas = [prop[0] for prop in arrow_properties]
        sizes = [prop[1] for prop in arrow_properties]
        angles = [prop[2] for prop in arrow_properties]

        logger.info("\nArrow Statistics:")
        logger.info(f"Average area: {np.mean(areas):.1f}")
        logger.info(f"Average width: {np.mean([s[0] for s in sizes]):.1f}")
        logger.info(f"Average height: {np.mean([s[1] for s in sizes]):.1f}")
        logger.info(f"Angle range: {min(angles):.1f} to {max(angles):.1f}")

if __name__ == "__main__":
    main()