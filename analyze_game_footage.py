"""Script to analyze both gameplay videos for arrow properties"""
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('GameFootageAnalysis')

def analyze_image_colors(image_path):
    """Analyze colors in the static game image"""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define multiple color ranges to find the arrow
    color_ranges = [
        ((80, 150, 150), (120, 255, 255)),  # Bright blue
        ((90, 160, 180), (110, 200, 255)),  # Current range
        ((85, 170, 200), (115, 210, 255))   # Another blue variant
    ]

    for i, (lower, upper) in enumerate(color_ranges):
        lower = np.array(lower)
        upper = np.array(upper)

        # Create mask and find contours
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.info(f"\nTesting color range {i+1}:")
        logger.info(f"Lower: {lower}")
        logger.info(f"Upper: {upper}")

        if contours:
            # Analyze each contour
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2 <= area <= 30:  # Reasonable size for arrow
                    # Get the color of the detected area
                    mask_single = np.zeros_like(mask)
                    cv2.drawContours(mask_single, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(hsv, mask=mask_single)[:3]

                    logger.info(f"Found potential arrow:")
                    logger.info(f"Area: {area}")
                    logger.info(f"Mean HSV: {mean_color}")

                    # Draw result
                    result = image.copy()
                    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 1)
                    cv2.imwrite(f'arrow_detection_{i}.png', result)

def analyze_video(video_path, description):
    """Analyze a gameplay video for arrow properties"""
    logger.info(f"\nAnalyzing {description}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    arrow_properties = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Test multiple color ranges
            color_ranges = [
                ((95, 165, 190), (115, 250, 255)),  # Previous range
                ((90, 160, 180), (120, 255, 255)),  # Wider range
                ((85, 150, 170), (125, 255, 255))   # Even wider range
            ]

            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 2 <= area <= 50:  # Expanded range for higher resolution
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.array(box, dtype=np.int32)

                        # Get the color of the detected area
                        mask_single = np.zeros_like(mask)
                        cv2.drawContours(mask_single, [cnt], -1, 255, -1)
                        mean_color = cv2.mean(hsv, mask=mask_single)[:3]

                        # Save properties
                        arrow_properties.append({
                            'frame': frame_count,
                            'area': area,
                            'width': rect[1][0],
                            'height': rect[1][1],
                            'angle': rect[2],
                            'color': mean_color
                        })

                        # Save visualization for first few detections
                        if len(arrow_properties) <= 5:
                            result = frame.copy()
                            cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
                            cv2.imwrite(f'arrow_detection_{description}_{len(arrow_properties)}.png', result)

        frame_count += 1
        if frame_count >= 100:  # Analyze first 100 frames
            break

    cap.release()

    if arrow_properties:
        areas = [prop['area'] for prop in arrow_properties]
        widths = [prop['width'] for prop in arrow_properties]
        heights = [prop['height'] for prop in arrow_properties]
        angles = [prop['angle'] for prop in arrow_properties]
        colors = [prop['color'] for prop in arrow_properties]

        logger.info(f"\nArrow Statistics for {description}:")
        logger.info(f"Area range: {min(areas):.1f} to {max(areas):.1f}")
        logger.info(f"Average dimensions: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
        logger.info(f"Angle range: {min(angles):.1f} to {max(angles):.1f}")
        logger.info("Color ranges (HSV):")
        logger.info(f"H: {min(c[0] for c in colors):.1f} to {max(c[0] for c in colors):.1f}")
        logger.info(f"S: {min(c[1] for c in colors):.1f} to {max(c[1] for c in colors):.1f}")
        logger.info(f"V: {min(c[2] for c in colors):.1f} to {max(c[2] for c in colors):.1f}")

def main():
    # Analyze both videos
    analyze_video('attached_assets/25.02.2025_10.10.43_REC.mp4', 'original_video')
    analyze_video('attached_assets/25.02.2025_11.19.54_REC.mp4', 'high_res_video')

if __name__ == "__main__":
    main()