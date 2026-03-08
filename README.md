# AI-Powered Driver Emotion and Stress Monitoring System

Real-time facial emotion and stress detection using webcam with either:
- **DeepFace backend** (default) for plug-and-play emotion analysis
- **Custom Keras model backend** (DenseNet-based), reused from `main.py`

## Features

### Emotion Detection
- **Real-time Detection** - Processes webcam feed and detects emotions frame-by-frame
- **8 Emotion Classes** - Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Backends**:
  - **DeepFace** – default, no training required
  - **Custom Keras model** – your own `.keras` model (DenseNet-based)
- **Visual Feedback** - Color-coded bounding boxes with probability bars
- **Statistics Tracking** - Session-wide emotion detection statistics
- **Screenshot Capture** - Save frames on demand
- **Frame Saving** - Auto-save all detected emotion frames

### Stress Monitoring & Safety Features
- **Real-time Stress Detection** - Calculates stress levels based on emotion analysis
- **4 Stress Levels** - LOW, MODERATE, HIGH, CRITICAL with color-coded indicators
- **High Stress Warnings** - Visual alerts when stress exceeds safe thresholds
- **Safety Stop System** - Automatic safety stop activation on critical stress levels
- **Stress Statistics** - Comprehensive stress level tracking and reporting
- **Driver Safety Analysis** - Detailed safety report at session end

## Requirements

```
tensorflow          # for custom Keras backend
opencv-python
numpy
deepface            # for DeepFace backend
simpleaudio         # for audio alerts (cross-platform)
mediapipe           # for drowsiness (eye/face landmarks)
```

## Usage

### Basic Usage (DeepFace backend, default)

```bash
python main.py
```

### Using Custom Keras Model Backend

```bash
python main.py --backend custom --model best_emotion_model.keras
```

### CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--backend` | `-b` | `deepface` | Backend to use: `deepface` or `custom` |
| `--model` | `-m` | `best_emotion_model.keras` | Path to the trained Keras model file (used when `--backend custom`) |
| `--camera` | `-c` | `0` | Camera device ID |
| `--save-frames` | `-s` | `False` | Save frames with detected emotions (`detected_emotions/` or `detected_emotions_deepface/`) |
| `--verbose` | `-v` | `False` | Enable verbose/debug logging |
| `--no-safety-stop` | | `False` | Disable automatic safety stop on critical stress |
| `--no-audio-alerts` | | `False` | Disable audio alerts for stress and drowsiness |

### Examples

```bash
# Use default settings (DeepFace backend)
python main.py

# Use DeepFace with a different camera
python main.py --camera 1

# Save all detected emotion frames (DeepFace backend)
python main.py --save-frames

# Use a custom Keras model backend
python main.py --backend custom --model path/to/custom_model.keras

# Use a different Keras model and disable safety stop
python main.py --backend custom --model final_emotion_model.keras --no-safety-stop

# Disable audio alerts (run silently)
python main.py --no-audio-alerts

# Enable verbose logging
python main.py -v

# Combine multiple options
python main.py -c 1 -s -v
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Take screenshot |
| `R` | Reset statistics |
| `P` | Pause/Resume detection |
| `T` | Toggle safety stop feature |

## Stress Detection System

The system calculates stress levels based on detected emotions using weighted scoring:

- **Angry** (0.9), **Fear** (0.85), **Sad** (0.75) - High stress contributors
- **Disgust** (0.7), **Contempt** (0.65) - Moderate stress contributors
- **Surprise** (0.4) - Variable stress indicator
- **Neutral** (0.1) - Low stress
- **Happy** (-0.3) - Stress reducer

### Stress Levels

- **LOW** (0-30%) - Green indicator, safe driving
- **MODERATE** (30-60%) - Yellow indicator, monitor closely
- **HIGH** (60-85%) - Orange indicator, warning displayed
- **CRITICAL** (85-100%) - Red indicator, safety stop activated after 3 seconds

### Safety Stop Feature

When critical stress is detected for more than 3 seconds:
- Visual alert with red border flashing
- Window title changes to indicate safety stop
- Statistics track safety events
- Can be toggled on/off with `T` key

## Output

When you quit the application, a comprehensive statistics summary is displayed:

```
============================================================
AI-POWERED DRIVER EMOTION & STRESS MONITORING REPORT
============================================================

Total detections: 82
Total stress readings: 82

EMOTION DETECTION STATISTICS:
------------------------------------------------------------
Fear       | █████████████████    |   72 ( 87.8%)
Happy      | ██                   |   10 ( 12.2%)
...

STRESS LEVEL STATISTICS:
------------------------------------------------------------
🟢 LOW      | ████████████████     |   45 ( 54.9%)
🟡 MODERATE | ████████             |   20 ( 24.4%)
🟠 HIGH     | ████                 |   12 ( 14.6%)
🔴 CRITICAL | █                    |    5 (  6.1%)

SAFETY ANALYSIS:
------------------------------------------------------------
⚠️  High/Critical stress events: 17
⚠️  Safety stop triggered: Yes
============================================================
```
