"""
AI-Powered Driver Emotion and Stress Monitoring System
Real-time facial emotion and stress detection for accident prevention.
Uses webcam feed with DenseNet-based model for facial emotion recognition.
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
import time
from deepface import DeepFace
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pygame
    # Force pygame to hide its welcome message
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
except ImportError:
    pygame = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Emotion labels (FERPlus 8-class standard)
CLASS_LABELS = {
    0: 'Angry', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
    4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Surprise'
}

# Emotion color mapping (BGR format)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),       # Red
    'Contempt': (128, 0, 128),  # Purple
    'Disgust': (0, 128, 0),     # Dark Green
    'Fear': (0, 165, 255),      # Orange
    'Happy': (0, 255, 0),       # Green
    'Neutral': (255, 200, 0),   # Cyan
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (0, 255, 255)   # Yellow
}

# Detection parameters
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction
INPUT_SIZE = (48, 48)

# Stress detection parameters
STRESS_EMOTION_WEIGHTS = {
    'Angry': 0.9,
    'Fear': 0.85,
    'Sad': 0.75,
    'Disgust': 0.7,
    'Contempt': 0.65,
    'Surprise': 0.4,  # Can be positive or negative
    'Neutral': 0.1,
    'Happy': -0.3  # Reduces stress
}

STRESS_LEVELS = {
    'LOW': (0.0, 0.3),
    'MODERATE': (0.3, 0.6),
    'HIGH': (0.6, 0.85),
    'CRITICAL': (0.85, 1.0)
}

STRESS_COLORS = {
    'LOW': (0, 255, 0),        # Green
    'MODERATE': (0, 255, 255),  # Yellow
    'HIGH': (0, 165, 255),      # Orange
    'CRITICAL': (0, 0, 255)     # Red
}

# Safety parameters
HIGH_STRESS_THRESHOLD = 0.7  # Trigger alert at this stress level
CRITICAL_STRESS_THRESHOLD = 0.85  # Trigger safety stop at this level
STRESS_WINDOW_SIZE = 30  # Number of frames to average stress over
HIGH_STRESS_DURATION_THRESHOLD = 3.0  # Seconds of high stress before stop


class DrowsinessState:
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    ASLEEP = "ASLEEP"


class AudioAlertManager:
    """
    Cross-platform audio alert helper using pygame when available.
    Plays short beeps for:
      - High/critical stress
      - Drowsiness / sleeping
    """

    def __init__(
        self,
        enabled: bool = True,
        min_interval_high_stress: float = 2.0,
        min_interval_drowsy: float = 2.0,
        custom_audio_high: str | None = None,
        custom_audio_drowsy: str | None = None,
    ) -> None:
        self.enabled = enabled and (pygame is not None)
        self.min_interval_high_stress = min_interval_high_stress
        self.min_interval_drowsy = min_interval_drowsy
        self._last_high_stress_play = 0.0
        self._last_drowsy_play = 0.0

        self._sound_high = None
        self._sound_drowsy = None

        if self.enabled:
            # Initialize pygame mixer safely
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
            except Exception as e:
                logger.warning(f"Pygame mixer initialization failed: {e}")
                self.enabled = False
                return

            # Default to alarm_audio.mp3 if it exists in current dir and arguments weren't provided
            default_alarm_file = "alarm_audio.mp3"
            
            if custom_audio_high is None and os.path.exists(default_alarm_file):
                custom_audio_high = default_alarm_file
            
            if custom_audio_drowsy is None and os.path.exists(default_alarm_file):
                custom_audio_drowsy = default_alarm_file

            # 1. Try loading custom audio for high stress
            if custom_audio_high and os.path.exists(custom_audio_high):
                try:
                    self._sound_high = pygame.mixer.Sound(custom_audio_high)
                    logger.info(f"Loaded custom high-stress audio from {custom_audio_high}")
                except Exception as e:
                    logger.error(f"Failed to load custom high-stress audio: {e}. Falling back to default.")

            # 2. Try loading custom audio for drowsiness
            if custom_audio_drowsy and os.path.exists(custom_audio_drowsy):
                try:
                    self._sound_drowsy = pygame.mixer.Sound(custom_audio_drowsy)
                    logger.info(f"Loaded custom drowsy audio from {custom_audio_drowsy}")
                except Exception as e:
                    logger.error(f"Failed to load custom drowsy audio: {e}. Falling back to default.")

            # 3. Generate defaults if custom loading failed or wasn't provided
            try:
                import wave
                import tempfile
                sample_rate = 44100
                duration_s = 0.3
                t = np.linspace(0, duration_s, int(sample_rate * duration_s), False)

                if self._sound_high is None:
                    tone_high = 0.8 * np.sin(2 * np.pi * 880 * t)
                    audio_high = np.int16(tone_high * 32767)
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        with wave.open(f, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            wf.writeframes(audio_high.tobytes())
                        temp_path_high = f.name
                    
                    self._sound_high = pygame.mixer.Sound(temp_path_high)
                    os.remove(temp_path_high)

                if self._sound_drowsy is None:
                    tone_drowsy = 0.6 * np.sin(2 * np.pi * 440 * t)
                    audio_drowsy = np.int16(tone_drowsy * 32767)
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        with wave.open(f, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sample_rate)
                            wf.writeframes(audio_drowsy.tobytes())
                        temp_path_drowsy = f.name
                        
                    self._sound_drowsy = pygame.mixer.Sound(temp_path_drowsy)
                    os.remove(temp_path_drowsy)
                    
            except Exception as e:
                logger.warning(f"Audio initialization failed, disabling alerts: {e}")
                self.enabled = False

    def _can_play(self, last_play_time: float, min_interval: float) -> bool:
        now = time.time()
        return now - last_play_time >= min_interval

    def play_high_stress_alert(self) -> None:
        """Play an alert sound for high/critical stress."""
        if not self.enabled or self._sound_high is None:
            return
        if not self._can_play(self._last_high_stress_play, self.min_interval_high_stress):
            return
        try:
            self._sound_high.play()
            self._last_high_stress_play = time.time()
        except Exception as e:
            logger.debug(f"High-stress audio alert failed: {e}")

    def play_drowsy_alert(self) -> None:
        """Play an alert sound when driver is drowsy/sleeping."""
        if not self.enabled or self._sound_drowsy is None:
            return
        if not self._can_play(self._last_drowsy_play, self.min_interval_drowsy):
            return
        try:
            self._sound_drowsy.play()
            self._last_drowsy_play = time.time()
        except Exception as e:
            logger.debug(f"Drowsy audio alert failed: {e}")


class DrowsinessDetector:
    """
    Drowsiness detector based on eye aspect ratio (EAR) from MediaPipe face mesh landmarks.
    """

    def __init__(
        self,
        ear_drowsy_threshold: float = 0.23,
        ear_sleep_threshold: float = 0.18,
    ) -> None:
        self.ear_drowsy_threshold = ear_drowsy_threshold
        self.ear_sleep_threshold = ear_sleep_threshold

        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            logger.info("Downloading face_landmarker.task model for MediaPipe...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Failed to download MediaPipe model: {e}")

        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe FaceLandmarker: {e}")
            self._detector = None

        # Approximate landmark indices for eyes (MediaPipe FaceMesh topology)
        self._right_eye_idx = [33, 160, 158, 133, 153, 144]
        self._left_eye_idx = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def _euclidean_dist(p1, p2) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _eye_aspect_ratio(self, eye_points) -> float:
        p1, p2, p3, p4, p5, p6 = eye_points
        v1 = self._euclidean_dist(p2, p6)
        v2 = self._euclidean_dist(p3, p5)
        h = self._euclidean_dist(p1, p4)
        if h <= 1e-6:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def analyze_frame(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        """
        Analyze a BGR frame and return (state, confidence).
        """
        if not hasattr(self, '_detector') or self._detector is None:
            return DrowsinessState.AWAKE, 0.0

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        
        if not result.face_landmarks:
            return DrowsinessState.AWAKE, 0.0

        face_landmarks = result.face_landmarks[0]

        def get_points(idx_list):
            pts = []
            for idx in idx_list:
                lm = face_landmarks[idx]
                pts.append((lm.x * width, lm.y * height))
            return pts

        right_eye_pts = get_points(self._right_eye_idx)
        left_eye_pts = get_points(self._left_eye_idx)

        ear_right = self._eye_aspect_ratio(right_eye_pts)
        ear_left = self._eye_aspect_ratio(left_eye_pts)
        ear = (ear_right + ear_left) / 2.0

        if ear <= self.ear_sleep_threshold:
            state = DrowsinessState.ASLEEP
        elif ear <= self.ear_drowsy_threshold:
            state = DrowsinessState.DROWSY
        else:
            state = DrowsinessState.AWAKE

        # For now, treat confidence as 1.0 when not AWAKE
        confidence = 1.0 if state != DrowsinessState.AWAKE else 0.0
        return state, confidence


class EmotionDetector:
    """AI-powered driver emotion and stress monitoring system for accident prevention."""

    def __init__(self, model_path: str, camera_id: int = 0, save_frames: bool = False,
                 enable_safety_stop: bool = True, audio_manager: AudioAlertManager | None = None):
        """
        Initialize the emotion and stress detector.

        Args:
            model_path: Path to the trained Keras model file
            camera_id: Camera device ID (default: 0)
            save_frames: Whether to save detected emotion frames
            enable_safety_stop: Enable automatic safety stop on critical stress
        """
        self.model_path = Path(model_path)
        self.camera_id = camera_id
        self.save_frames = save_frames
        self.enable_safety_stop = enable_safety_stop
        self.audio_manager = audio_manager
        self.model = None
        self.cap = None
        self.face_cascade = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = datetime.now()

        # Statistics tracking
        self.emotion_counts = {label: 0 for label in CLASS_LABELS.values()}
        self.total_detections = 0

        # Stress monitoring
        self.stress_history = deque(maxlen=STRESS_WINDOW_SIZE)
        self.current_stress_level = 0.0
        self.stress_level_label = 'LOW'
        self.high_stress_start_time = None
        self.safety_stop_active = False
        self.stress_level_counts = {level: 0 for level in STRESS_LEVELS.keys()}
        self.total_stress_readings = 0

        # High-stress audio timing
        self.high_stress_audio_start_time = None

        # Drowsiness audio timing
        self.drowsy_audio_start_time = None
        self.asleep_audio_start_time = None

        # Model compatibility info
        self.model_input_size = INPUT_SIZE  # Will be updated after model load
        self.model_num_classes = len(CLASS_LABELS)  # Will be updated after model load

        # Drowsiness detection
        self.drowsiness_detector = DrowsinessDetector()
        self.drowsiness_state = DrowsinessState.AWAKE
        self.drowsiness_confidence = 0.0
        self.drowsy_start_time = None
        self.asleep_start_time = None

        # Output directory for saved frames
        if save_frames:
            self.output_dir = Path("detected_emotions")
            self.output_dir.mkdir(exist_ok=True)

    def load_model(self) -> bool:
        """Load the pre-trained emotion detection model."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False

            logger.info(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            
            # Validate model compatibility
            if not self._validate_model():
                return False
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            logger.info(f"Number of output classes: {self.model.output_shape[-1]}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _validate_model(self) -> bool:
        """Validate that the model is compatible with the detection system."""
        try:
            # Check if model has input and output
            if self.model is None:
                logger.error("Model is None")
                return False
            
            # Check input shape
            input_shape = self.model.input_shape
            if input_shape is None or len(input_shape) < 3:
                logger.error(f"Invalid model input shape: {input_shape}")
                return False
            
            # Check output shape
            output_shape = self.model.output_shape
            if output_shape is None or len(output_shape) < 1:
                logger.error(f"Invalid model output shape: {output_shape}")
                return False
            
            num_classes = output_shape[-1]
            if num_classes != len(CLASS_LABELS):
                logger.warning(f"Model has {num_classes} classes but expected {len(CLASS_LABELS)} classes")
                logger.warning("The model may use different emotion labels. Proceeding with caution...")
            
            # Test prediction with dummy data
            test_input_shape = input_shape[1:]  # Remove batch dimension
            if len(test_input_shape) == 3:
                h, w, c = test_input_shape
                test_input = np.zeros((1, h, w, c), dtype=np.float32)
            else:
                logger.error(f"Unexpected input shape format: {input_shape}")
                return False
            
            try:
                test_output = self.model.predict(test_input, verbose=0)
                if test_output is None or len(test_output.shape) < 2:
                    logger.error(f"Model prediction returned invalid output shape: {test_output.shape if test_output is not None else None}")
                    return False
                
                # Store model info for later use
                if len(input_shape) >= 3:
                    self.model_input_size = (input_shape[2], input_shape[1])  # (width, height)
                self.model_num_classes = output_shape[-1]
                
                logger.info("Model validation successful!")
                return True
            except Exception as pred_error:
                logger.error(f"Model prediction test failed: {pred_error}")
                import traceback
                logger.debug(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return False

    def initialize_camera(self) -> bool:
        """Initialize webcam and face detection cascade."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                return False

            logger.info("Camera initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def preprocess_face(self, gray_frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Preprocess a detected face region for model input.

        Args:
            gray_frame: Grayscale frame
            x, y, w, h: Face bounding box coordinates

        Returns:
            Preprocessed face array ready for model prediction
        """
        try:
            # Extract and resize ROI
            roi = gray_frame[y:y+h, x:x+w]
            
            # Get expected input size from model
            if hasattr(self, 'model_input_size') and self.model_input_size:
                target_size = self.model_input_size
            elif self.model and self.model.input_shape:
                model_input_shape = self.model.input_shape[1:3]  # Get height, width (skip batch and channels)
                target_size = (model_input_shape[1], model_input_shape[0])  # (width, height) for cv2.resize
            else:
                target_size = INPUT_SIZE
            
            roi = cv2.resize(roi, target_size)

            # Check if model expects RGB or grayscale
            if self.model and self.model.input_shape:
                expected_channels = self.model.input_shape[-1]
                if expected_channels == 3:
                    # Convert to RGB (DenseNet expects 3 channels)
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
                elif expected_channels == 1:
                    # Keep grayscale, add channel dimension
                    roi = np.expand_dims(roi, axis=-1)
            else:
                # Default: convert to RGB
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

            # Normalize and add batch dimension
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)

            return roi
        except Exception as e:
            logger.debug(f"Preprocessing error: {e}")
            # Fallback to default preprocessing
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, INPUT_SIZE)
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)
            return roi

    def predict_emotion(self, face_roi: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Predict emotion from preprocessed face region.

        Args:
            face_roi: Preprocessed face array

        Returns:
            Tuple of (emotion_label, confidence, all_predictions)
        """
        try:
            # Get predictions
            pred_output = self.model.predict(face_roi, verbose=0)
            
            # Handle different output formats
            if isinstance(pred_output, list):
                predictions = pred_output[0]
            else:
                predictions = pred_output
            
            # Handle batch dimension
            if len(predictions.shape) > 1:
                predictions = predictions[0]
            
            # Ensure predictions is a numpy array
            predictions = np.array(predictions).flatten()
            
            # Validate number of classes
            num_classes = len(predictions)
            expected_classes = len(CLASS_LABELS)
            
            if num_classes != expected_classes:
                logger.warning(f"Model output has {num_classes} classes but expected {expected_classes}")
                # If model has fewer classes, pad with zeros
                if num_classes < expected_classes:
                    predictions = np.pad(predictions, (0, expected_classes - num_classes), 'constant')
                # If model has more classes, truncate
                elif num_classes > expected_classes:
                    predictions = predictions[:expected_classes]
                    logger.warning(f"Truncated predictions from {num_classes} to {expected_classes} classes")
            
            # Get emotion with highest confidence
            emotion_idx = int(np.argmax(predictions))
            
            # Ensure index is valid
            if emotion_idx >= len(CLASS_LABELS):
                emotion_idx = emotion_idx % len(CLASS_LABELS)
                logger.warning(f"Emotion index {emotion_idx} out of range, using {emotion_idx % len(CLASS_LABELS)}")
            
            emotion_label = CLASS_LABELS[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            # Ensure confidence is valid
            if np.isnan(confidence) or np.isinf(confidence):
                confidence = 0.0
                logger.warning("Invalid confidence value, setting to 0.0")

            return emotion_label, confidence, predictions
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return default values on error
            return 'Neutral', 0.0, np.zeros(len(CLASS_LABELS))

    def calculate_stress_level(self, emotion: str, confidence: float, predictions: np.ndarray) -> float:
        """
        Calculate stress level based on detected emotions.

        Args:
            emotion: Detected emotion label
            confidence: Prediction confidence
            predictions: All emotion predictions

        Returns:
            Stress level (0.0 to 1.0)
        """
        try:
            # Calculate weighted stress score from all emotions
            stress_score = 0.0
            total_prob = 0.0

            for idx, label in CLASS_LABELS.items():
                weight = STRESS_EMOTION_WEIGHTS.get(label, 0.0)
                prob = float(predictions[idx])
                
                # Add weighted contribution (positive weights increase stress, negative decrease it)
                stress_score += weight * prob
                total_prob += prob

            # Normalize stress score
            if total_prob > 0.001:  # Avoid division by very small numbers
                normalized_stress = stress_score / total_prob
            else:
                normalized_stress = 0.0

            # Map normalized stress to [0, 1] range
            # Since weights range from -0.3 (Happy) to 0.9 (Angry)
            # normalized_stress will range approximately from -0.3 to 0.9
            # We need to map this to [0, 1]
            # Formula: (value - min) / (max - min) = (value + 0.3) / 1.2
            stress_score = (normalized_stress + 0.3) / 1.2
            
            # Clamp to [0, 1] range to ensure valid values
            stress_score = max(0.0, min(1.0, stress_score))

            # Add to history for smoothing
            self.stress_history.append(stress_score)

            # Calculate average stress over window
            if len(self.stress_history) > 0:
                avg_stress = float(np.mean(list(self.stress_history)))
            else:
                avg_stress = stress_score

            # Ensure valid float (handle NaN or Inf)
            if np.isnan(avg_stress) or np.isinf(avg_stress):
                avg_stress = 0.0

            return float(avg_stress)
        except Exception as e:
            logger.debug(f"Error calculating stress level: {e}")
            return 0.0

    def get_stress_level_label(self, stress_level: float) -> str:
        """Get stress level label based on stress value."""
        for level, (low, high) in STRESS_LEVELS.items():
            if low <= stress_level < high:
                return level
        return 'CRITICAL'  # Default for values >= 1.0

    def check_safety_stop(self, stress_level: float) -> bool:
        """
        Check if safety stop should be triggered based on stress level.

        Args:
            stress_level: Current stress level

        Returns:
            True if safety stop should be triggered
        """
        if not self.enable_safety_stop:
            return False

        current_time = time.time()

        if stress_level >= CRITICAL_STRESS_THRESHOLD:
            if self.high_stress_start_time is None:
                self.high_stress_start_time = current_time
            elif current_time - self.high_stress_start_time >= HIGH_STRESS_DURATION_THRESHOLD:
                return True
        else:
            self.high_stress_start_time = None

        return False

    def handle_high_stress_audio(self, stress_level: float) -> None:
        """
        Trigger high-stress audio alert when stress has been high/critical
        for a continuous duration.
        """
        if self.audio_manager is None or not self.audio_manager.enabled:
            return

        now = time.time()

        if stress_level >= HIGH_STRESS_THRESHOLD:
            if self.high_stress_audio_start_time is None:
                self.high_stress_audio_start_time = now
                return

            duration = now - self.high_stress_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_high_stress_alert()
        else:
            self.high_stress_audio_start_time = None

    def handle_drowsiness_audio(self, state: str) -> None:
        """
        Trigger drowsiness audio alert when driver is drowsy/asleep
        for a continuous duration.
        """
        if self.audio_manager is None or not self.audio_manager.enabled:
            return

        now = time.time()

        if state == DrowsinessState.DROWSY:
            if self.drowsy_audio_start_time is None:
                self.drowsy_audio_start_time = now
                return
            duration = now - self.drowsy_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_drowsy_alert()
        else:
            self.drowsy_audio_start_time = None

        if state == DrowsinessState.ASLEEP:
            if self.asleep_audio_start_time is None:
                self.asleep_audio_start_time = now
                return
            duration = now - self.asleep_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_drowsy_alert()
        else:
            self.asleep_audio_start_time = None

    def draw_detection(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                       emotion: str, confidence: float, predictions: np.ndarray,
                       stress_level: float = None) -> None:
        """
        Draw detection results on the frame including stress information.

        Args:
            frame: Video frame to draw on
            x, y, w, h: Face bounding box coordinates
            emotion: Detected emotion label
            confidence: Prediction confidence
            predictions: All emotion predictions for probability bars
            stress_level: Current stress level (optional)
        """
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Draw face bounding box with thickness based on stress
        box_thickness = 2
        if stress_level is not None:
            if stress_level >= CRITICAL_STRESS_THRESHOLD:
                box_thickness = 5
            elif stress_level >= HIGH_STRESS_THRESHOLD:
                box_thickness = 4

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, box_thickness)

        # Draw emotion label with confidence
        label_text = f"{emotion} {int(confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Background for label
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw stress level if available
        if stress_level is not None:
            stress_label = self.get_stress_level_label(stress_level)
            stress_color = STRESS_COLORS.get(stress_label, (255, 255, 255))
            stress_text = f"Stress: {stress_label} ({int(stress_level * 100)}%)"
            stress_size, _ = cv2.getTextSize(stress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for stress label
            cv2.rectangle(frame, (x, y + h + 2), (x + stress_size[0] + 10, y + h + 25), stress_color, -1)
            cv2.putText(frame, stress_text, (x + 5, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw emotion probability bars (mini visualization)
        bar_start_y = y + h + (30 if stress_level is not None else 5)
        bar_height = 8
        bar_width = w

        for i, (idx, label) in enumerate(CLASS_LABELS.items()):
            if i >= 4:  # Only show top 4 for space
                break
            prob = predictions[idx]
            bar_length = int(prob * bar_width)
            bar_y = bar_start_y + (i * (bar_height + 2))

            # Bar background
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Bar fill
            bar_color = EMOTION_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x, bar_y), (x + bar_length, bar_y + bar_height), bar_color, -1)

    def draw_hud(self, frame: np.ndarray) -> None:
        """Draw heads-up display with FPS, stress level, drowsiness, and safety status."""
        height, width = frame.shape[:2]

        # Semi-transparent overlay for HUD
        overlay = frame.copy()

        # FPS display
        cv2.putText(overlay, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detection count
        cv2.putText(overlay, f"Detections: {self.total_detections}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Stress level display
        stress_color = STRESS_COLORS.get(self.stress_level_label, (255, 255, 255))
        stress_text = f"Stress: {self.stress_level_label} ({int(self.current_stress_level * 100)}%)"
        cv2.putText(overlay, stress_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2)

        # Stress level bar
        bar_x, bar_y = 10, 95
        bar_width, bar_height = 200, 20
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        bar_fill_width = int(self.current_stress_level * bar_width)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_fill_width, bar_y + bar_height), stress_color, -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Drowsiness display
        drowsy_text = f"Drowsiness: {self.drowsiness_state}"
        drowsy_color = (0, 255, 0)
        if self.drowsiness_state == DrowsinessState.DROWSY:
            drowsy_color = (0, 255, 255)
        elif self.drowsiness_state == DrowsinessState.ASLEEP:
            drowsy_color = (0, 0, 255)
        cv2.putText(overlay, drowsy_text, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsy_color, 2)

        # Safety status
        y_offset = 155
        if self.safety_stop_active:
            cv2.putText(overlay, "SAFETY STOP ACTIVE!", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            # Blinking effect
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 10)
        elif self.current_stress_level >= HIGH_STRESS_THRESHOLD:
            warning_text = f"WARNING: High Stress Detected!"
            cv2.putText(overlay, warning_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Controls help
        controls = [
            "Q - Quit",
            "S - Screenshot",
            "R - Reset Stats",
            "P - Pause",
            "T - Toggle Safety Stop"
        ]
        for i, ctrl in enumerate(controls):
            cv2.putText(overlay, ctrl, (width - 150, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Blend overlay
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    def calculate_fps(self) -> None:
        """Calculate and update FPS."""
        self.frame_count += 1
        now = datetime.now()
        elapsed = (now - self.last_fps_time).total_seconds()

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

    def save_detection(self, frame: np.ndarray, emotion: str) -> None:
        """Save frame with detected emotion."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"{emotion}_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.debug(f"Saved detection: {filename}")

    def print_statistics(self) -> None:
        """Print detection statistics summary including stress analysis."""
        print("\n" + "=" * 60)
        print("AI-POWERED DRIVER EMOTION & STRESS MONITORING REPORT")
        print("=" * 60)

        if self.total_detections == 0:
            print("No emotions detected during this session.")
            return

        print(f"\nTotal detections: {self.total_detections}")
        print(f"Total stress readings: {self.total_stress_readings}\n")

        # Emotion statistics
        print("EMOTION DETECTION STATISTICS:")
        print("-" * 60)
        sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)

        for emotion, count in sorted_emotions:
            percentage = (count / self.total_detections) * 100
            bar = "█" * int(percentage / 5)
            print(f"{emotion:10s} | {bar:20s} | {count:4d} ({percentage:5.1f}%)")

        # Stress statistics
        if self.total_stress_readings > 0:
            print("\nSTRESS LEVEL STATISTICS:")
            print("-" * 60)
            sorted_stress = sorted(self.stress_level_counts.items(), 
                                 key=lambda x: STRESS_LEVELS[x[0]][0])

            for level, count in sorted_stress:
                percentage = (count / self.total_stress_readings) * 100
                bar = "█" * int(percentage / 5)
                color_indicator = "🔴" if level == "CRITICAL" else "🟠" if level == "HIGH" else "🟡" if level == "MODERATE" else "🟢"
                print(f"{color_indicator} {level:10s} | {bar:20s} | {count:4d} ({percentage:5.1f}%)")

            # Safety events
            print("\nSAFETY ANALYSIS:")
            print("-" * 60)
            high_stress_count = self.stress_level_counts.get('HIGH', 0) + self.stress_level_counts.get('CRITICAL', 0)
            if high_stress_count > 0:
                print(f"⚠️  High/Critical stress events: {high_stress_count}")
                print(f"⚠️  Safety stop triggered: {'Yes' if self.safety_stop_active else 'No'}")
            else:
                print("✅ No high stress events detected - Safe driving session")

        print("=" * 60)

    def run(self) -> None:
        """Main detection loop."""
        if not self.load_model():
            return

        if not self.initialize_camera():
            return

        logger.info("Starting AI-powered driver emotion and stress monitoring system...")
        logger.info("Press 'Q' to quit, 'T' to toggle safety stop")
        paused = False

        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to capture frame")
                        break

                    frame = cv2.flip(frame, 1)  # Mirror effect
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Drowsiness analysis (per frame)
                    try:
                        state, confidence = self.drowsiness_detector.analyze_frame(frame)
                        self.drowsiness_state = state
                        self.drowsiness_confidence = confidence
                    except Exception as d_err:
                        logger.debug(f"Drowsiness analysis error: {d_err}")

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=FACE_SCALE_FACTOR,
                        minNeighbors=FACE_MIN_NEIGHBORS,
                        minSize=(30, 30)
                    )

                    # Process each detected face
                    for (x, y, w, h) in faces:
                        try:
                            face_roi = self.preprocess_face(gray, x, y, w, h)
                            emotion, confidence, predictions = self.predict_emotion(face_roi)

                            if confidence >= CONFIDENCE_THRESHOLD:
                                # Calculate stress level (with error handling)
                                try:
                                    stress_level = self.calculate_stress_level(emotion, confidence, predictions)
                                    self.current_stress_level = stress_level
                                    self.stress_level_label = self.get_stress_level_label(stress_level)
                                    
                                    # Update stress statistics
                                    self.stress_level_counts[self.stress_level_label] += 1
                                    self.total_stress_readings += 1

                                    # Check for safety stop
                                    if self.check_safety_stop(stress_level):
                                        self.safety_stop_active = True
                                        logger.warning(f"SAFETY STOP ACTIVATED - Critical stress detected: {stress_level:.2f}")

                                    # High-stress audio alert
                                    self.handle_high_stress_audio(stress_level)

                                    # Draw detection with stress information
                                    self.draw_detection(frame, x, y, w, h, emotion, confidence, predictions, stress_level)
                                except Exception as stress_error:
                                    # If stress calculation fails, still draw emotion detection
                                    logger.debug(f"Stress calculation error: {stress_error}")
                                    self.draw_detection(frame, x, y, w, h, emotion, confidence, predictions, None)

                                # Update statistics
                                self.emotion_counts[emotion] += 1
                                self.total_detections += 1

                                # Save frame if enabled
                                if self.save_frames:
                                    self.save_detection(frame, emotion)
                        except Exception as e:
                            logger.warning(f"Face processing error: {e}", exc_info=True)
                            continue

                    # Update HUD
                    self.calculate_fps()
                    # Drowsiness audio based on latest state
                    self.handle_drowsiness_audio(self.drowsiness_state)
                    self.draw_hud(frame)

                    # Display frame
                    window_title = 'Driver Emotion & Stress Monitor'
                    cv2.imshow(window_title, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
                    logger.info(f"Screenshot saved: screenshot_{timestamp}.jpg")
                elif key == ord('r'):
                    # Reset statistics
                    self.emotion_counts = {label: 0 for label in CLASS_LABELS.values()}
                    self.total_detections = 0
                    logger.info("Statistics reset")
                elif key == ord('p'):
                    # Toggle pause
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")
                elif key == ord('t'):
                    # Toggle safety stop
                    self.enable_safety_stop = not self.enable_safety_stop
                    if not self.enable_safety_stop:
                        self.safety_stop_active = False
                    logger.info(f"Safety stop {'enabled' if self.enable_safety_stop else 'disabled'}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources and print final statistics."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.print_statistics()
        logger.info("Cleanup complete")


class DeepFaceDriverMonitor:
    """Driver monitoring using DeepFace for emotion + stress, with safety stop."""

    def __init__(
        self,
        camera_id: int = 0,
        analysis_interval: float = 1.0,
        save_frames: bool = False,
        enable_safety_stop: bool = True,
        audio_manager: AudioAlertManager | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.analysis_interval = analysis_interval
        self.save_frames = save_frames
        self.enable_safety_stop = enable_safety_stop
        self.audio_manager = audio_manager

        self.cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = datetime.now()

        # Stats
        self.emotion_counts = {label: 0 for label in CLASS_LABELS.values()}
        self.total_detections = 0

        # Stress
        self.stress_history = deque(maxlen=STRESS_WINDOW_SIZE)
        self.current_stress_level = 0.0
        self.stress_level_label = "LOW"
        self.high_stress_start_time: float | None = None
        self.safety_stop_active = False
        self.stress_level_counts = {level: 0 for level in STRESS_LEVELS.keys()}
        self.total_stress_readings = 0

        # High-stress audio timing
        self.high_stress_audio_start_time: float | None = None

        # Drowsiness audio timing
        self.drowsy_audio_start_time: float | None = None
        self.asleep_audio_start_time: float | None = None

        # Drowsiness detection
        self.drowsiness_detector = DrowsinessDetector()
        self.drowsiness_state = DrowsinessState.AWAKE
        self.drowsiness_confidence = 0.0
        self.drowsy_start_time: float | None = None
        self.asleep_start_time: float | None = None

        # DeepFace state
        self.last_analysis_time = 0.0
        self.current_emotion = "Analyzing..."
        self.current_confidence = 0.0
        self.current_predictions = np.zeros(len(CLASS_LABELS), dtype=np.float32)
        self.current_region: dict | None = None

        # Output directory
        if save_frames:
            self.output_dir = Path("detected_emotions_deepface")
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None

    def _map_deepface_emotions(self, df_emotions: dict) -> np.ndarray:
        """Map DeepFace emotion dict to our CLASS_LABELS vector."""
        key_map = {
            "angry": "Angry",
            "disgust": "Disgust",
            "fear": "Fear",
            "fearful": "Fear",
            "happy": "Happy",
            "sad": "Sad",
            "surprise": "Surprise",
            "surprised": "Surprise",
            "neutral": "Neutral",
            "contempt": "Contempt",
        }

        scores = {label: 0.0 for label in CLASS_LABELS.values()}

        for k, v in df_emotions.items():
            try:
                mapped = key_map.get(k.lower())
                if mapped is not None and mapped in scores:
                    scores[mapped] = float(v)
            except Exception:
                continue

        vec = np.array([scores[label] for label in CLASS_LABELS.values()], dtype=np.float32)
        total = float(vec.sum())
        if total > 1e-6:
            vec /= total
        return vec

    def calculate_stress_level(self, predictions: np.ndarray) -> float:
        """Weighted stress calculation using predictions vector."""
        try:
            stress_score = 0.0
            total_prob = 0.0
            for idx, label in CLASS_LABELS.items():
                weight = STRESS_EMOTION_WEIGHTS.get(label, 0.0)
                prob = float(predictions[idx])
                stress_score += weight * prob
                total_prob += prob

            if total_prob > 0.001:
                normalized_stress = stress_score / total_prob
            else:
                normalized_stress = 0.0

            stress_score = (normalized_stress + 0.3) / 1.2
            stress_score = max(0.0, min(1.0, stress_score))

            self.stress_history.append(stress_score)
            if self.stress_history:
                avg_stress = float(np.mean(list(self.stress_history)))
            else:
                avg_stress = stress_score

            if np.isnan(avg_stress) or np.isinf(avg_stress):
                avg_stress = 0.0

            return avg_stress
        except Exception as e:
            logger.debug(f"Error calculating stress level (DeepFace): {e}")
            return 0.0

    def get_stress_level_label(self, stress_level: float) -> str:
        for level, (low, high) in STRESS_LEVELS.items():
            if low <= stress_level < high:
                return level
        return "CRITICAL"

    def check_safety_stop(self, stress_level: float) -> bool:
        if not self.enable_safety_stop:
            return False

        now = time.time()
        if stress_level >= CRITICAL_STRESS_THRESHOLD:
            if self.high_stress_start_time is None:
                self.high_stress_start_time = now
            elif now - self.high_stress_start_time >= HIGH_STRESS_DURATION_THRESHOLD:
                return True
        else:
            self.high_stress_start_time = None
        return False

    def handle_high_stress_audio(self, stress_level: float) -> None:
        """
        Trigger high-stress audio alert when stress has been high/critical
        for a continuous duration.
        """
        if self.audio_manager is None or not self.audio_manager.enabled:
            return

        now = time.time()

        if stress_level >= HIGH_STRESS_THRESHOLD:
            if self.high_stress_audio_start_time is None:
                self.high_stress_audio_start_time = now
                return

            duration = now - self.high_stress_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_high_stress_alert()
        else:
            self.high_stress_audio_start_time = None

    def handle_drowsiness_audio(self, state: str) -> None:
        """
        Trigger drowsiness audio alert when driver is drowsy/asleep
        for a continuous duration.
        """
        if self.audio_manager is None or not self.audio_manager.enabled:
            return

        now = time.time()

        if state == DrowsinessState.DROWSY:
            if self.drowsy_audio_start_time is None:
                self.drowsy_audio_start_time = now
                return
            duration = now - self.drowsy_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_drowsy_alert()
        else:
            self.drowsy_audio_start_time = None

        if state == DrowsinessState.ASLEEP:
            if self.asleep_audio_start_time is None:
                self.asleep_audio_start_time = now
                return
            duration = now - self.asleep_audio_start_time
            if duration >= HIGH_STRESS_DURATION_THRESHOLD:
                self.audio_manager.play_drowsy_alert()
        else:
            self.asleep_audio_start_time = None

    def draw_detection(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        """Draw bounding box + labels using current emotion/stress state."""
        emotion = self.current_emotion
        confidence = self.current_confidence
        predictions = self.current_predictions
        stress_level = self.current_stress_level

        color = EMOTION_COLORS.get(emotion.capitalize(), (255, 255, 255))

        box_thickness = 2
        if stress_level >= CRITICAL_STRESS_THRESHOLD:
            box_thickness = 5
        elif stress_level >= HIGH_STRESS_THRESHOLD:
            box_thickness = 4

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

        label_text = f"{emotion.capitalize()} {int(confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(
            frame,
            label_text,
            (x + 5, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        stress_label = self.stress_level_label
        stress_color = STRESS_COLORS.get(stress_label, (255, 255, 255))
        stress_text = f"Stress: {stress_label} ({int(stress_level * 100)}%)"
        stress_size, _ = cv2.getTextSize(stress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x, y + h + 2),
            (x + stress_size[0] + 10, y + h + 25),
            stress_color,
            -1,
        )
        cv2.putText(
            frame,
            stress_text,
            (x + 5, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        bar_start_y = y + h + 30
        bar_height = 8
        bar_width = w
        for i, (idx, label) in enumerate(CLASS_LABELS.items()):
            if i >= 4:
                break
            prob = float(predictions[idx])
            bar_length = int(prob * bar_width)
            bar_y = bar_start_y + i * (bar_height + 2)
            cv2.rectangle(
                frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1
            )
            bar_color = EMOTION_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(
                frame,
                (x, bar_y),
                (x + bar_length, bar_y + bar_height),
                bar_color,
                -1,
            )

    def draw_hud(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.putText(
            overlay,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            overlay,
            f"Detections: {self.total_detections}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        stress_color = STRESS_COLORS.get(self.stress_level_label, (255, 255, 255))
        stress_text = f"Stress: {self.stress_level_label} ({int(self.current_stress_level * 100)}%)"
        cv2.putText(
            overlay,
            stress_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            stress_color,
            2,
        )

        bar_x, bar_y = 10, 95
        bar_width, bar_height = 200, 20
        cv2.rectangle(
            overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1
        )
        bar_fill_width = int(self.current_stress_level * bar_width)
        cv2.rectangle(
            overlay,
            (bar_x, bar_y),
            (bar_x + bar_fill_width, bar_y + bar_height),
            stress_color,
            -1,
        )
        cv2.rectangle(
            overlay,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (255, 255, 255),
            2,
        )

        # Drowsiness display
        drowsy_text = f"Drowsiness: {self.drowsiness_state}"
        drowsy_color = (0, 255, 0)
        if self.drowsiness_state == DrowsinessState.DROWSY:
            drowsy_color = (0, 255, 255)
        elif self.drowsiness_state == DrowsinessState.ASLEEP:
            drowsy_color = (0, 0, 255)
        cv2.putText(
            overlay,
            drowsy_text,
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            drowsy_color,
            2,
        )

        y_offset = 155
        if self.safety_stop_active:
            cv2.putText(
                overlay,
                "SAFETY STOP ACTIVE!",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 10)
        elif self.current_stress_level >= HIGH_STRESS_THRESHOLD:
            cv2.putText(
                overlay,
                "WARNING: High Stress Detected!",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        controls = [
            "Q - Quit",
            "S - Screenshot",
            "R - Reset Stats",
            "P - Pause",
            "T - Toggle Safety Stop",
        ]
        for i, ctrl in enumerate(controls):
            cv2.putText(
                overlay,
                ctrl,
                (w - 200, 25 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    def calculate_fps(self) -> None:
        self.frame_count += 1
        now = datetime.now()
        elapsed = (now - self.last_fps_time).total_seconds()
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now

    def save_detection(self, frame: np.ndarray, emotion: str) -> None:
        if not self.output_dir:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"{emotion}_{ts}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.debug(f"Saved detection (DeepFace): {filename}")

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return

        logger.info(
            "Starting DeepFace-based driver emotion & stress monitor "
            "(backend=deepface). Press 'Q' to quit."
        )

        paused = False
        try:
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to capture frame")
                        break

                    frame = cv2.flip(frame, 1)
                    now = time.time()

                    # Drowsiness analysis (per frame)
                    try:
                        state, confidence = self.drowsiness_detector.analyze_frame(frame)
                        self.drowsiness_state = state
                        self.drowsiness_confidence = confidence
                    except Exception as d_err:
                        logger.debug(f"DeepFace drowsiness analysis error: {d_err}")

                    # Run DeepFace analysis at lower frequency
                    if now - self.last_analysis_time >= self.analysis_interval:
                        try:
                            analysis = DeepFace.analyze(
                                frame,
                                actions=["emotion"],
                                enforce_detection=False,
                            )
                            if isinstance(analysis, list):
                                analysis = analysis[0]

                            df_emotions = analysis.get("emotion", {})
                            region = analysis.get("region", None)
                            dominant = str(analysis.get("dominant_emotion", "neutral"))

                            preds = self._map_deepface_emotions(df_emotions)
                            self.current_predictions = preds

                            # Map dominant label to CLASS_LABELS index
                            dominant_key = dominant.lower()
                            if dominant_key == "surprised":
                                dominant_key = "surprise"
                            if dominant_key == "fearful":
                                dominant_key = "fear"

                            emotion_idx = 0
                            for idx, label in CLASS_LABELS.items():
                                if label.lower() == dominant_key:
                                    emotion_idx = idx
                                    break

                            self.current_emotion = CLASS_LABELS[emotion_idx]
                            self.current_confidence = float(preds[emotion_idx])

                            # Stress updates
                            stress_level = self.calculate_stress_level(preds)
                            self.current_stress_level = stress_level
                            self.stress_level_label = self.get_stress_level_label(stress_level)
                            self.stress_level_counts[self.stress_level_label] += 1
                            self.total_stress_readings += 1

                            if self.check_safety_stop(stress_level):
                                self.safety_stop_active = True
                                logger.warning(
                                    f"SAFETY STOP ACTIVATED - Critical stress: {stress_level:.2f}"
                                )

                            # High-stress audio alert
                            self.handle_high_stress_audio(stress_level)

                            self.total_detections += 1
                            self.emotion_counts[self.current_emotion] += 1

                            if region and all(k in region for k in ("x", "y", "w", "h")):
                                self.current_region = region
                            else:
                                self.current_region = None

                            self.last_analysis_time = now
                        except Exception as e:
                            logger.debug(f"DeepFace analysis error: {e}")

                    # Draw detection if we have a region
                    if self.current_region:
                        x = int(self.current_region.get("x", 0))
                        y = int(self.current_region.get("y", 0))
                        w = int(self.current_region.get("w", 100))
                        h = int(self.current_region.get("h", 100))
                        h_img, w_img = frame.shape[:2]
                        x = max(0, min(x, w_img - 1))
                        y = max(0, min(y, h_img - 1))
                        w = max(10, min(w, w_img - x))
                        h = max(10, min(h, h_img - y))
                        self.draw_detection(frame, x, y, w, h)
                    else:
                        cv2.putText(
                            frame,
                            f"Emotion: {self.current_emotion} ({int(self.current_confidence * 100)}%)",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )

                    self.calculate_fps()
                    # Drowsiness audio based on latest state
                    self.handle_drowsiness_audio(self.drowsiness_state)
                    self.draw_hud(frame)

                    title = "Driver Emotion & Stress Monitor (DeepFace)"
                    cv2.imshow(title, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user")
                    break
                elif key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"deepface_screenshot_{ts}.jpg", frame)
                    logger.info(f"Screenshot saved: deepface_screenshot_{ts}.jpg")
                elif key == ord("r"):
                    self.emotion_counts = {label: 0 for label in CLASS_LABELS.values()}
                    self.total_detections = 0
                    self.stress_level_counts = {level: 0 for level in STRESS_LEVELS.keys()}
                    self.total_stress_readings = 0
                    self.stress_history.clear()
                    logger.info("Statistics reset")
                elif key == ord("p"):
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")
                elif key == ord("t"):
                    self.enable_safety_stop = not self.enable_safety_stop
                    if not self.enable_safety_stop:
                        self.safety_stop_active = False
                    logger.info(
                        f"Safety stop {'enabled' if self.enable_safety_stop else 'disabled'}"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("DeepFace driver monitor cleanup complete")


def main():
    """CLI entrypoint with backend selection: deepface | custom."""
    parser = argparse.ArgumentParser(
        description=(
            "AI-Powered Driver Emotion & Stress Monitoring (DeepFace or Custom Model)\n"
            "- DeepFace backend: uses DeepFace for emotion, with stress & safety stop.\n"
            "- Custom backend: uses Keras model via EmotionDetector."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Q - Quit the application
  S - Take a screenshot
  R - Reset statistics
  P - Pause/Resume detection
  T - Toggle safety stop feature

Safety Features:
  - Real-time stress level monitoring
  - High stress warnings
  - Automatic safety stop on critical stress levels
  - Driver safety statistics and reporting

Examples:
  python main.py                           # DeepFace backend (default)
  python main.py --backend deepface        # Explicit DeepFace backend
  python main.py --backend custom --model best_emotion_model.keras
  python main.py --backend custom --model final_emotion_model.keras --no-safety-stop
        """,
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["deepface", "custom"],
        default="deepface",
        help="Which backend to use: 'deepface' (default) or 'custom' Keras model",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="best_emotion_model.keras",
        help="Path to the trained Keras model file (used only when backend='custom')",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--save-frames",
        "-s",
        action="store_true",
        help="Save frames with detected emotions to disk",
    )
    parser.add_argument(
        "--no-safety-stop",
        action="store_true",
        help="Disable automatic safety stop on critical stress",
    )
    parser.add_argument(
        "--no-audio-alerts",
        action="store_true",
        help="Disable audio alerts for stress and drowsiness",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--audio-high",
        type=str,
        default=None,
        help="Path to a custom .wav file for high stress auto alerts",
    )
    parser.add_argument(
        "--audio-drowsy",
        type=str,
        default=None,
        help="Path to a custom .wav file for drowsiness auto alerts",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    audio_manager = AudioAlertManager(
        enabled=not args.no_audio_alerts,
        custom_audio_high=args.audio_high,
        custom_audio_drowsy=args.audio_drowsy,
    )

    if args.backend == "deepface":
        monitor = DeepFaceDriverMonitor(
            camera_id=args.camera,
            analysis_interval=1.0,
            save_frames=args.save_frames,
            enable_safety_stop=not args.no_safety_stop,
            audio_manager=audio_manager,
        )
        monitor.run()
    else:
        detector = EmotionDetector(
            model_path=args.model,
            camera_id=args.camera,
            save_frames=args.save_frames,
            enable_safety_stop=not args.no_safety_stop,
            audio_manager=audio_manager,
        )
        detector.run()


if __name__ == "__main__":
    main()