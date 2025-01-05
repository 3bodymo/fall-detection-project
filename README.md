# Fall Detection Project

This project implements a fall detection system using YOLO for person detection, a person tracker, and an LSTM-based fall detector.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. You have to download a YOLO model, and we recommend you to use this model `yolo11l.pt`:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -P models/
   ```
4. Update the model path in `config.py`:
   ```python
   YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11l.pt')
   ```

## Usage

### FallDetectionSystem

The `FallDetectionSystem` class provides methods to process video frames and detect falls.

#### Initialization

To initialize the fall detection system:

```python
from fall_detection_system import FallDetectionSystem

fall_detection_system = FallDetectionSystem()
```

#### Process a Single Frame

To process a single frame and detect falls:

```python
# ...existing code to open video file and set up video writer...

processed_frame, has_fall = fall_detection_system.process_frame(frame)

# ...existing code to display the processed frame...
```

#### Process a Video

To process a video file and save the annotated video with fall detection results:

```python
video_path = "path_to_input_video.mp4"
output_path = "path_to_output_video.mp4"

fall_detection_system.process_video(video_path, output_path)
```

Alternatively, you can change the `video_path` and `output_path` variables directly in `fall_detection_system.py` and run the file.
