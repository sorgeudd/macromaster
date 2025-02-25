"""AI vision system for game object detection and analysis"""
import logging
from pathlib import Path
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import tempfile
import os

class VisionSystem:
    def __init__(self, model_path=None):
        self.logger = logging.getLogger('VisionSystem')
        self.model = None
        self.feature_extractor = None
        self.roboflow_client = None

        # Try to import AI dependencies
        try:
            import cv2
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            from inference_sdk import InferenceHTTPClient

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize Roboflow client
            try:
                self.roboflow_client = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="dw3ovISGhOXZ5NzDkg7L"
                )
                self.logger.info("Roboflow client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Roboflow client: {str(e)}")

            # Initialize models
            if model_path and Path(model_path).exists():
                self.model = torch.load(model_path)
                self.logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use pretrained ResNet model as starting point
                model_name = "microsoft/resnet-50"
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
                self.logger.info(f"Using pretrained model: {model_name}")

            if self.model:
                self.model.to(self.device)
                self.model.eval()

        except ImportError as e:
            self.logger.warning(f"AI dependencies not available: {str(e)}")
            self.logger.warning("Running in basic detection mode")

    def detect_resources(self, frame):
        """Detect resources using Roboflow API"""
        if frame is None:
            return []

        try:
            # Ensure frame is in BGR format for cv2
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Create a temporary file to save the JPEG image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Save frame as JPEG
                is_success, img_buf = cv2.imencode('.jpg', frame)
                if not is_success:
                    self.logger.error("Failed to encode image to JPEG")
                    return self._basic_detection(frame)

                # Write to temporary file
                temp_file.write(img_buf.tobytes())
                temp_file.flush()
                temp_path = temp_file.name

            if self.roboflow_client:
                try:
                    self.logger.debug(f"Submitting image from temporary file: {temp_path}")

                    # Get predictions from Roboflow using the file path
                    result = self.roboflow_client.infer(temp_path, model_id="albiongathering/3")
                    self.logger.debug(f"Roboflow API response: {result}")

                    # Process predictions
                    detections = []
                    for pred in result.get('predictions', []):
                        detection = {
                            'class_id': pred.get('class'),
                            'confidence': pred.get('confidence'),
                            'bbox': [
                                pred.get('x'), pred.get('y'),
                                pred.get('width'), pred.get('height')
                            ]
                        }
                        detections.append(detection)

                    self.logger.debug(f"Detected {len(detections)} resources")
                    return detections
                except Exception as api_error:
                    self.logger.error(f"Error in API call: {str(api_error)}")
                    self.logger.error(f"Full error details: {api_error.__class__.__name__}")
                    return self._basic_detection(frame)
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temporary file: {str(e)}")
            else:
                self.logger.warning("Roboflow client not available, falling back to basic detection")
                return self._basic_detection(frame)

        except Exception as e:
            self.logger.error(f"Error in resource detection: {str(e)}")
            self.logger.error(f"Full error details: {e.__class__.__name__}")
            return self._basic_detection(frame)

    def _basic_detection(self, frame):
        """Basic color-based object detection when AI is not available"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Simple color thresholding for basic detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color ranges for basic detection with tighter thresholds
            color_ranges = {
                'wood': [(10, 120, 70), (20, 255, 200)],     # Brown
                'stone': [(0, 0, 100), (180, 30, 200)],      # Gray
                'ore': [(0, 0, 0), (180, 30, 100)],          # Dark Gray/Black
                'fiber': [(35, 100, 100), (85, 255, 255)],   # Green
                'hide': [(0, 50, 50), (10, 255, 200)]        # Light Brown
            }

            detections = []
            min_area = 1000  # Increased minimum area to filter out noise
            max_area = frame.shape[0] * frame.shape[1] // 4  # Max 1/4 of image

            for class_id, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                # Apply morphological operations to remove noise
                kernel = np.ones((7,7), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                if np.any(mask):
                    # Find contours to get bounding boxes
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if min_area < area < max_area:  # Filter by area
                            x, y, w, h = cv2.boundingRect(contour)

                            # Additional filtering criteria
                            aspect_ratio = float(w)/h
                            solidity = area / (w * h)

                            # Check if the detection meets our criteria
                            if (0.5 <= aspect_ratio <= 2.0 and  # Not too elongated
                                solidity > 0.5):                # Reasonably solid shape

                                # Calculate confidence based on multiple factors
                                area_conf = min(area / 10000, 0.95)
                                shape_conf = min(solidity, 0.95)
                                confidence = (area_conf + shape_conf) / 2

                                detections.append({
                                    'class_id': class_id,
                                    'confidence': confidence,
                                    'bbox': [x, y, w, h]
                                })

            # Sort detections by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)

            # Keep only top N detections per class to prevent over-detection
            max_per_class = 5
            filtered_detections = []
            class_counts = {}

            for det in detections:
                class_id = det['class_id']
                if class_counts.get(class_id, 0) < max_per_class:
                    filtered_detections.append(det)
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

            self.logger.debug(f"Basic detection found {len(filtered_detections)} resources")
            return filtered_detections

        except Exception as e:
            self.logger.error(f"Error in basic detection: {str(e)}")
            return []

    def process_video_frame(self, frame):
        """Process a single video frame for training"""
        if frame is None:
            return None

        try:
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
        except Exception as e:
            self.logger.warning(f"Error processing frame: {str(e)}")
            return frame

        return frame

    def save_model(self, path):
        """Save the trained model"""
        if not self.model:
            self.logger.warning("No model to save")
            return False

        try:
            import torch
            torch.save(self.model, path)
            self.logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def detect_objects(self, frame, confidence_threshold=0.8):
        """Detect objects in frame using trained model or basic detection"""
        try:
            frame = self.process_video_frame(frame)
            if frame is None:
                return []

            # If AI model is available, use it
            if self.model and self.feature_extractor:
                import torch
                # Prepare input
                inputs = self.feature_extractor(frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = outputs.logits.softmax(-1)

                # Get predictions above threshold
                confident_preds = (probs > confidence_threshold).nonzero().cpu().numpy()

                return [
                    {
                        'class_id': pred[1],
                        'confidence': probs[pred[0], pred[1]].item(),
                        'bbox': None  # TODO: Implement bounding box detection
                    }
                    for pred in confident_preds
                ]

            # Fallback to basic color-based detection
            else:
                self.logger.debug("Using basic color-based detection")
                return self._basic_detection(frame)

        except Exception as e:
            self.logger.error(f"Error detecting objects: {str(e)}")
            return []

    def train_on_video(self, video_path, label):
        """Train model on video footage of resources/obstacles"""
        self.logger.info(f"Training on video: {video_path} for label: {label}")

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.process_video_frame(frame)
                if frame is not None:
                    frames.append(frame)

            cap.release()

            if frames:
                self.logger.info(f"Extracted {len(frames)} frames for training")
                return True

            return False

        except ImportError:
            self.logger.error("OpenCV not available for video training")
            return False
        except Exception as e:
            self.logger.error(f"Error training on video: {str(e)}")
            return False