# Emotion Detection

Real-time facial emotion detection using webcam with a DenseNet-based deep learning model.

## Features

- **Real-time Detection** - Processes webcam feed and detects emotions frame-by-frame
- **8 Emotion Classes** - Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Visual Feedback** - Color-coded bounding boxes with probability bars
- **Statistics Tracking** - Session-wide emotion detection statistics
- **Screenshot Capture** - Save frames on demand
- **Frame Saving** - Auto-save all detected emotion frames

## Requirements

```
tensorflow
opencv-python
numpy
```

## Usage

### Basic Usage

```bash
python main.py
```

### CLI Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | `best_emotion_model.keras` | Path to the trained Keras model file |
| `--camera` | `-c` | `0` | Camera device ID |
| `--save-frames` | `-s` | `False` | Save frames with detected emotions to `detected_emotions/` |
| `--verbose` | `-v` | `False` | Enable verbose/debug logging |

### Examples

```bash
# Use default settings
python main.py

# Use a different camera
python main.py --camera 1

# Save all detected emotion frames
python main.py --save-frames

# Use a custom model
python main.py --model path/to/custom_model.keras

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

## Output

When you quit the application, a statistics summary is displayed:

```
==================================================
EMOTION DETECTION STATISTICS
==================================================
Total detections: 82

Fear       | █████████████████    |   72 ( 87.8%)
Happy      | ██                   |   10 ( 12.2%)
...
==================================================
```
