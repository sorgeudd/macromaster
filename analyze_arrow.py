"""Script to analyze arrow orientations from video footage"""
import cv2
import numpy as np

def analyze_video_frames(video_path):
    """Extract and analyze frames from video to study arrow orientations"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 30 == 0:  # Analyze every 30th frame
            # Save frame for analysis
            cv2.imwrite(f'arrow_frame_{frame_count}.png', frame)
            
            # Convert to HSV for arrow detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Yellow arrow mask
            lower = np.array([25, 100, 150])
            upper = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Save mask for analysis
            cv2.imwrite(f'arrow_mask_{frame_count}.png', mask)
            
        frame_count += 1
    
    cap.release()
    print(f"Analyzed {frame_count} frames, saved {frame_count//30} snapshots")

if __name__ == "__main__":
    analyze_video_frames("attached_assets/25.02.2025_12.55.00_REC.mp4")
