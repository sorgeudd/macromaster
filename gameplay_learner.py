"""AI system for learning and replicating player gameplay patterns"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

@dataclass
class GameplayPattern:
    """Base class for all gameplay patterns"""
    count: int = 0
    success_rate: float = 0.0
    total_time: float = 0.0

    def update(self, success_rate: float, time_taken: float = 0.0):
        self.count += 1
        self.total_time += time_taken
        self.success_rate = (self.success_rate * (self.count - 1) + success_rate) / self.count

    def to_dict(self) -> Dict:
        """Convert pattern to dictionary for serialization"""
        return {
            'count': self.count,
            'success_rate': self.success_rate,
            'total_time': self.total_time
        }

    def reset(self):
        """Reset pattern to initial state"""
        self.count = 0
        self.success_rate = 0.0
        self.total_time = 0.0

@dataclass
class GameplayAction:
    """Represents a single player action"""
    action_type: str  # 'move', 'gather', 'combat', 'mount', 'cast', 'reel', 'timeout'
    timestamp: float
    position: Tuple[int, int]
    target_position: Optional[Tuple[int, int]] = None
    resource_type: Optional[str] = None
    combat_ability: Optional[str] = None
    success_rate: float = 1.0
    success: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert action to dictionary for serialization"""
        return {
            'action_type': self.action_type,
            'timestamp': self.timestamp,
            'position': self.position,
            'target_position': self.target_position,
            'resource_type': self.resource_type,
            'combat_ability': self.combat_ability,
            'success_rate': self.success_rate,
            'success': self.success,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GameplayAction':
        """Create action from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

class GameplayLearner:
    def __init__(self):
        self.logger = logging.getLogger('GameplayLearner')
        self.recorded_actions: List[GameplayAction] = []
        self.movement_patterns: Dict[Tuple[Tuple[int, int], Tuple[int, int]], GameplayPattern] = {}
        self.resource_preferences: Dict[str, GameplayPattern] = {}
        self.combat_patterns: Dict[str, GameplayPattern] = {}
        self.learning_start_time: Optional[float] = None
        self.is_learning: bool = False

        # Initialize AI models as None first
        self.timing_predictor: Optional[nn.Sequential] = None
        self.success_predictor: Optional[nn.Sequential] = None

        # Initialize AI components
        self._init_ai_models()

    def import_video_for_training(self, video_path: str) -> bool:
        """Import and analyze video file for training
        Args:
            video_path: Path to the video file
        Returns:
            bool: True if video was successfully processed
        """
        try:
            self.logger.info(f"Starting video analysis: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error("Error opening video file")
                return False

            # Start learning mode
            self.start_learning()
            frame_count = 0
            last_frame = None
            last_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_time = time.time()

                # Process frame every 100ms to detect actions
                if current_time - last_time >= 0.1:
                    if last_frame is not None:
                        # Detect movement
                        movement = self._detect_movement(last_frame, frame)
                        if movement:
                            self.record_action('move', movement['position'], 
                                             target_position=movement['target'])

                        # Detect combat
                        combat = self._detect_combat(frame)
                        if combat:
                            self.record_action('combat', combat['position'],
                                             combat_ability=combat['ability'])

                        # Detect resource gathering
                        resource = self._detect_resource_gathering(frame)
                        if resource:
                            self.record_action('gather', resource['position'],
                                             resource_type=resource['type'])

                    last_frame = frame.copy()
                    last_time = current_time

            # Stop learning and analyze patterns
            self.stop_learning()
            cap.release()

            self.logger.info(f"Completed video analysis: {frame_count} frames processed")
            return True

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            if cap:
                cap.release()
            return False

    def _detect_movement(self, prev_frame, curr_frame) -> Optional[Dict]:
        """Detect movement between frames"""
        try:
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                              0.5, 3, 15, 3, 5, 1.2, 0)

            # Get magnitude and angle of 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # If significant movement detected
            if np.mean(magnitude) > 1.0:
                # Calculate average movement direction
                dx = np.mean(flow[..., 0])
                dy = np.mean(flow[..., 1])

                # Get current position (center of frame)
                h, w = curr_frame.shape[:2]
                curr_pos = (w//2, h//2)

                # Calculate target position based on movement
                target_pos = (int(w//2 + dx*10), int(h//2 + dy*10))

                return {
                    'position': curr_pos,
                    'target': target_pos
                }

            return None

        except Exception as e:
            self.logger.error(f"Error detecting movement: {str(e)}")
            return None

    def _detect_combat(self, frame) -> Optional[Dict]:
        """Detect combat actions in frame"""
        try:
            # Convert frame to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color ranges for combat effects (red for damage, etc.)
            combat_colors = [
                ((0, 100, 100), (10, 255, 255)),   # Red
                ((170, 100, 100), (180, 255, 255)), # Red (wrap-around)
            ]

            for lower, upper in combat_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if np.sum(mask) > 1000:  # If significant combat effect detected
                    # Find center of combat effect
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return {
                            'position': (cx, cy),
                            'ability': 'attack'  # Default to basic attack
                        }

            return None

        except Exception as e:
            self.logger.error(f"Error detecting combat: {str(e)}")
            return None

    def _detect_resource_gathering(self, frame) -> Optional[Dict]:
        """Detect resource gathering actions in frame"""
        try:
            # Convert frame to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color ranges for different resources
            resource_colors = {
                'fish': ((100, 100, 100), (130, 255, 255)),  # Blue for water
                'ore': ((0, 0, 100), (180, 30, 255)),        # Gray for rocks
                'wood': ((20, 100, 100), (30, 255, 255))     # Brown for trees
            }

            for resource_type, (lower, upper) in resource_colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if np.sum(mask) > 5000:  # If significant resource area detected
                    # Find center of resource area
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return {
                            'position': (cx, cy),
                            'type': resource_type
                        }

            return None

        except Exception as e:
            self.logger.error(f"Error detecting resources: {str(e)}")
            return None

    def reset_learning(self):
        """Reset all learned patterns"""
        try:
            self.logger.info("Resetting all learned patterns")

            # Clear recorded actions
            self.recorded_actions.clear()

            # Reset all pattern dictionaries
            self.movement_patterns.clear()
            self.resource_preferences.clear()
            self.combat_patterns.clear()

            # Save empty patterns
            self.save_patterns()

            self.logger.info("Successfully reset all learned patterns")
            return True

        except Exception as e:
            self.logger.error(f"Error resetting patterns: {str(e)}")
            return False

    def _init_ai_models(self):
        """Initialize machine learning models"""
        self.logger.info("Initializing machine learning models")
        try:
            # Initialize neural networks for pattern recognition
            self.timing_predictor = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

            self.success_predictor = nn.Sequential(
                nn.Linear(15, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

            # Load existing patterns if available
            self._load_patterns()

        except Exception as e:
            self.logger.warning(f"AI models not available - running in basic mode: {str(e)}")
            self.timing_predictor = None
            self.success_predictor = None

    def start_learning(self):
        """Start recording player actions"""
        self.learning_start_time = time.time()
        self.is_learning = True
        self.recorded_actions.clear()
        self.logger.info("Started learning mode")

    def stop_learning(self):
        """Stop recording and analyze patterns"""
        try:
            if not self.is_learning:
                self.logger.debug("Learning mode was not active")
                return True

            self.is_learning = False
            duration = time.time() - self.learning_start_time if self.learning_start_time else 0
            self.logger.info(f"Stopped learning mode after {duration:.1f} seconds")

            # Analyze recorded actions
            self._analyze_patterns()

            # Create models directory if it doesn't exist
            Path("models").mkdir(exist_ok=True)

            # Save learned patterns
            if not self.save_patterns():
                self.logger.error("Failed to save patterns after learning")
                return False

            self.logger.info("Successfully stopped learning mode and saved patterns")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping learning mode: {str(e)}")
            return False

    def record_action(self, action_type: str, position: Tuple[int, int], **kwargs):
        """Record a player action during learning mode"""
        if not self.is_learning:
            return

        try:
            action = GameplayAction(
                action_type=action_type,
                timestamp=time.time(),
                position=position,
                **{k: v for k, v in kwargs.items() if k in GameplayAction.__dataclass_fields__}
            )
            self.recorded_actions.append(action)
            self.logger.debug(f"Recorded action: {action_type} at {position}")
        except Exception as e:
            self.logger.error(f"Error recording action: {str(e)}")

    def _analyze_patterns(self):
        """Analyze recorded actions to learn patterns"""
        if not self.recorded_actions:
            return

        # Analyze movement patterns
        moves = [a for a in self.recorded_actions if a.action_type == 'move']
        if moves:
            self._analyze_movement(moves)

        # Analyze resource gathering
        gathers = [a for a in self.recorded_actions if a.action_type == 'gather']
        if gathers:
            self._analyze_gathering(gathers)

        # Analyze combat
        combat = [a for a in self.recorded_actions if a.action_type == 'combat']
        if combat:
            self._analyze_combat(combat)

        # Retrain models with new data if available
        if self.timing_predictor is not None and self.success_predictor is not None:
            self._train_timing_model()
            self._train_success_model()

    def _analyze_movement(self, moves: List[GameplayAction]):
        """Learn movement patterns"""
        for m1, m2 in zip(moves[:-1], moves[1:]):
            key = (m1.position, m2.position)
            if key not in self.movement_patterns:
                self.movement_patterns[key] = GameplayPattern()

            pattern = self.movement_patterns[key]
            time_taken = m2.timestamp - m1.timestamp
            pattern.update(m2.success_rate, time_taken)

    def _analyze_gathering(self, gathers: List[GameplayAction]):
        """Learn resource gathering patterns"""
        for action in gathers:
            if action.resource_type:
                if action.resource_type not in self.resource_preferences:
                    self.resource_preferences[action.resource_type] = GameplayPattern()

                pref = self.resource_preferences[action.resource_type]
                pref.update(action.success_rate)

    def _analyze_combat(self, combat: List[GameplayAction]):
        """Learn combat patterns"""
        for action in combat:
            if action.combat_ability:
                if action.combat_ability not in self.combat_patterns:
                    self.combat_patterns[action.combat_ability] = GameplayPattern()

                pattern = self.combat_patterns[action.combat_ability]
                pattern.update(action.success_rate)

    def _train_timing_model(self):
        """Train neural network for action timing prediction"""
        if self.timing_predictor is None:
            return

        try:
            # Prepare training data from patterns
            X = []  # Features
            y = []  # Target timing values

            for pattern in self.movement_patterns.values():
                if pattern.count > 0:
                    avg_time = pattern.total_time / pattern.count
                    features = self._extract_timing_features(pattern)
                    X.append(features)
                    y.append(avg_time)

            if X and y:
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

                # Train model
                optimizer = torch.optim.Adam(self.timing_predictor.parameters())
                criterion = nn.MSELoss()

                for _ in range(100):  # Training epochs
                    optimizer.zero_grad()
                    pred = self.timing_predictor(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()

        except Exception as e:
            self.logger.error(f"Error training timing model: {str(e)}")

    def _train_success_model(self):
        """Train neural network for success prediction"""
        if self.success_predictor is None:
            return

        try:
            # Prepare training data
            X = []  # Features
            y = []  # Success rates

            # Combine patterns from different action types
            all_patterns = {
                **self.movement_patterns,
                **self.resource_preferences,
                **self.combat_patterns
            }

            for pattern in all_patterns.values():
                if isinstance(pattern, GameplayPattern):
                    features = self._extract_success_features(pattern)
                    X.append(features)
                    y.append(pattern.success_rate)

            if X and y:
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

                # Train model
                optimizer = torch.optim.Adam(self.success_predictor.parameters())
                criterion = nn.BCELoss()

                for _ in range(100):  # Training epochs
                    optimizer.zero_grad()
                    pred = self.success_predictor(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()

        except Exception as e:
            self.logger.error(f"Error training success model: {str(e)}")

    def _extract_timing_features(self, pattern) -> List[float]:
        """Extract features for timing prediction"""
        if isinstance(pattern, GameplayPattern):
            return [
                pattern.count,
                pattern.total_time / max(1, pattern.count),
                pattern.success_rate,
                # Add more relevant features
            ] + [0] * 7  # Pad to 10 features
        return [0] * 10  # Return zeroed features for unsupported pattern types

    def _extract_success_features(self, pattern) -> List[float]:
        """Extract features for success prediction"""
        if isinstance(pattern, GameplayPattern):
            return [
                pattern.count,
                pattern.success_rate,
                # Add more relevant features
            ] + [0] * 13  # Pad to 15 features
        elif isinstance(pattern, dict):
            return [
                pattern.get('count', 0),
                pattern.get('success_rate', 0),
                # Add more relevant features
            ] + [0] * 13  # Pad to 15 features
        return [0] * 15  # Return zeroed features for unsupported pattern types

    def save_patterns(self) -> bool:
        """Public method to save learned patterns to file"""
        try:
            self._save_patterns()
            self.logger.info("Successfully saved patterns to file")
            return True
        except Exception as e:
            self.logger.error(f"Error saving patterns: {str(e)}")
            return False

    def _save_patterns(self):
        """Save learned patterns to file"""
        patterns = {
            'movement': {str(k): v.to_dict() for k, v in self.movement_patterns.items()},
            'resources': {k: v.to_dict() for k, v in self.resource_preferences.items()},
            'combat': {k: v.to_dict() for k, v in self.combat_patterns.items()}
        }

        pattern_file = Path("models/learned_patterns.json")
        pattern_file.parent.mkdir(exist_ok=True)

        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2)

        self.logger.info("Successfully saved patterns to file")


    def _load_patterns(self):
        """Load patterns from file and convert string keys back to tuples"""
        try:
            pattern_file = Path("models/learned_patterns.json")
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    patterns = json.load(f)

                # Convert string keys back to tuples for movement patterns
                self.movement_patterns = {}
                for k_str, v in patterns.get('movement', {}).items():
                    # Remove brackets and split coordinates
                    coords = k_str.strip('()').split('), (')
                    if len(coords) == 2:
                        pos1 = tuple(map(int, coords[0].split(', ')))
                        pos2 = tuple(map(int, coords[1].split(', ')))
                        key = (pos1, pos2)
                        pattern = GameplayPattern()
                        pattern.__dict__.update(v)
                        self.movement_patterns[key] = pattern

                # Load other patterns normally
                self.resource_preferences = {k: GameplayPattern() for k in patterns.get('resources', {})}
                for k, v in patterns.get('resources', {}).items():
                    self.resource_preferences[k].__dict__.update(v)

                self.combat_patterns = {k: GameplayPattern() for k in patterns.get('combat', {})}
                for k, v in patterns.get('combat', {}).items():
                    self.combat_patterns[k].__dict__.update(v)


                self.logger.info("Loaded learned patterns from file")

        except Exception as e:
            self.logger.error(f"Error loading patterns: {str(e)}")
            # Initialize empty patterns on error
            self.movement_patterns = {}
            self.resource_preferences = {}
            self.combat_patterns = {}

    def predict_next_action(self, current_state: Dict) -> Optional[Dict]:
        """Predict optimal next action based on learned patterns"""
        try:
            # If AI models aren't available, use basic pattern matching
            if not self.timing_predictor or not self.success_predictor:
                return self._basic_pattern_matching(current_state)

            # Extract current state features
            features = self._extract_state_features(current_state)

            # Get timing prediction
            timing_input = torch.tensor(features[:10], dtype=torch.float32)
            predicted_timing = self.timing_predictor(timing_input).item()

            # Get success prediction
            success_input = torch.tensor(features[:15], dtype=torch.float32)
            success_prob = self.success_predictor(success_input).item()

            # Find best matching pattern
            best_action = None
            best_score = -1

            for action_type in ['move', 'gather', 'combat']:
                patterns = self._get_patterns_for_type(action_type)
                for pattern in patterns:
                    score = self._score_pattern(pattern, current_state, success_prob)
                    if score > best_score:
                        best_score = score
                        best_action = {
                            'type': action_type,
                            'pattern': pattern,
                            'timing': predicted_timing,
                            'success_probability': success_prob
                        }

            return best_action

        except Exception as e:
            self.logger.error(f"Error predicting next action: {str(e)}")
            return self._basic_pattern_matching(current_state)

    def _basic_pattern_matching(self, current_state: Dict) -> Dict:
        """Basic pattern matching without AI models"""
        try:
            # Simple rule-based decision making
            if current_state.get('in_combat', False):
                return {
                    'type': 'combat',
                    'pattern': {'success_rate': 0.8},
                    'timing': 1.0,
                    'success_probability': 0.8
                }

            if current_state.get('detected_resources', []):
                return {
                    'type': 'gather',
                    'pattern': {'success_rate': 0.9},
                    'timing': 2.0,
                    'success_probability': 0.9
                }

            return {
                'type': 'move',
                'pattern': {'success_rate': 1.0},
                'timing': 0.5,
                'success_probability': 1.0
            }

        except Exception as e:
            self.logger.error(f"Error in basic pattern matching: {str(e)}")
            return {
                'type': 'move',
                'pattern': {'success_rate': 1.0},
                'timing': 1.0,
                'success_probability': 1.0
            }

    def _extract_state_features(self, state: Dict) -> List[float]:
        """Extract features from current game state"""
        return [
            state.get('health', 100) / 100,
            float(state.get('in_combat', False)),
            float(state.get('is_mounted', False)),
            len(state.get('detected_resources', [])),
            len(state.get('detected_obstacles', [])),
            # Add more state features
        ] + [0] * 10  # Pad to required size

    def _get_patterns_for_type(self, action_type: str) -> List[Dict]:
        """Get relevant patterns for action type"""
        if action_type == 'move':
            return [{'type': 'move', **vars(p)} for p in self.movement_patterns.values()]
        elif action_type == 'gather':
            return [{'type': 'gather', **p} for p in self.resource_preferences.values()]
        elif action_type == 'combat':
            return [{'type': 'combat', **vars(p)} for p in self.combat_patterns.values()]
        return []

    def _score_pattern(self, pattern: Dict, current_state: Dict, success_prob: float) -> float:
        """Score how well a pattern matches current state"""
        base_score = pattern.get('success_rate', 0) * success_prob

        # Add bonuses based on state
        if current_state.get('in_combat', False) and pattern['type'] == 'combat':
            base_score *= 1.5
        elif len(current_state.get('detected_resources', [])) > 0 and pattern['type'] == 'gather':
            base_score *= 1.3

        return base_score