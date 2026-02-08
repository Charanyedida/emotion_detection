"""
Real-time Emotion Detection Application
Uses webcam feed with DenseNet-based model for facial emotion recognition.
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import logging
from datetime import datetime
from pathlib import Path

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


class EmotionDetector:
    """Real-time emotion detection from webcam feed."""

    def __init__(self, model_path: str, camera_id: int = 0, save_frames: bool = False):
        """
        Initialize the emotion detector.

        Args:
            model_path: Path to the trained Keras model file
            camera_id: Camera device ID (default: 0)
            save_frames: Whether to save detected emotion frames
        """
        self.model_path = Path(model_path)
        self.camera_id = camera_id
        self.save_frames = save_frames
        self.model = None
        self.cap = None
        self.face_cascade = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = datetime.now()

        # Statistics tracking
        self.emotion_counts = {label: 0 for label in CLASS_LABELS.values()}
        self.total_detections = 0

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
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
        # Extract and resize ROI
        roi = gray_frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, INPUT_SIZE)

        # Convert to RGB (DenseNet expects 3 channels)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        # Normalize and add batch dimension
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
        predictions = self.model.predict(face_roi, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion_label = CLASS_LABELS[emotion_idx]
        confidence = float(predictions[emotion_idx])

        return emotion_label, confidence, predictions

    def draw_detection(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                       emotion: str, confidence: float, predictions: np.ndarray) -> None:
        """
        Draw detection results on the frame.

        Args:
            frame: Video frame to draw on
            x, y, w, h: Face bounding box coordinates
            emotion: Detected emotion label
            confidence: Prediction confidence
            predictions: All emotion predictions for probability bars
        """
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw emotion label with confidence
        label_text = f"{emotion} {int(confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Background for label
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw emotion probability bars (mini visualization)
        bar_start_y = y + h + 5
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
        """Draw heads-up display with FPS and controls."""
        height, width = frame.shape[:2]

        # Semi-transparent overlay for HUD
        overlay = frame.copy()

        # FPS display
        cv2.putText(overlay, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detection count
        cv2.putText(overlay, f"Detections: {self.total_detections}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Controls help
        controls = [
            "Q - Quit",
            "S - Screenshot",
            "R - Reset Stats",
            "P - Pause"
        ]
        for i, ctrl in enumerate(controls):
            cv2.putText(overlay, ctrl, (width - 130, 25 + i * 20),
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
        """Print detection statistics summary."""
        print("\n" + "=" * 50)
        print("EMOTION DETECTION STATISTICS")
        print("=" * 50)

        if self.total_detections == 0:
            print("No emotions detected during this session.")
            return

        print(f"Total detections: {self.total_detections}\n")

        # Sort emotions by count
        sorted_emotions = sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True)

        for emotion, count in sorted_emotions:
            percentage = (count / self.total_detections) * 100
            bar = "█" * int(percentage / 5)
            print(f"{emotion:10s} | {bar:20s} | {count:4d} ({percentage:5.1f}%)")

        print("=" * 50)

    def run(self) -> None:
        """Main detection loop."""
        if not self.load_model():
            return

        if not self.initialize_camera():
            return

        logger.info("Starting emotion detection... Press 'Q' to quit.")
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
                                self.draw_detection(frame, x, y, w, h, emotion, confidence, predictions)

                                # Update statistics
                                self.emotion_counts[emotion] += 1
                                self.total_detections += 1

                                # Save frame if enabled
                                if self.save_frames:
                                    self.save_detection(frame, emotion)
                        except Exception as e:
                            logger.debug(f"Face processing error: {e}")
                            continue

                    # Update HUD
                    self.calculate_fps()
                    self.draw_hud(frame)

                    # Display frame
                    cv2.imshow('Emotion Detector', frame)

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


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Real-time Emotion Detection using Webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Q - Quit the application
  S - Take a screenshot
  R - Reset statistics
  P - Pause/Resume detection

Examples:
  python main.py                           # Use default camera and model
  python main.py --camera 1                # Use camera ID 1
  python main.py --save-frames             # Save detected emotion frames
  python main.py --model custom_model.keras  # Use custom model
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='best_emotion_model.keras',
        help='Path to the trained Keras model file (default: best_emotion_model.keras)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--save-frames', '-s',
        action='store_true',
        help='Save frames with detected emotions to disk'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run detector
    detector = EmotionDetector(
        model_path=args.model,
        camera_id=args.camera,
        save_frames=args.save_frames
    )
    detector.run()


if __name__ == "__main__":
    main()