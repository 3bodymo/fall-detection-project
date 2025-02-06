# Fall Detection Project

This project implements a real-time fall detection system using YOLO for person detection, a person tracker, and an LSTM-based fall detector.

## Features

- Real-time person detection using YOLO
- Person tracking across frames
- Pose estimation and processing
- LSTM-based fall detection
- Support for both video files and live camera feed

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The system can be run using command-line arguments:

```bash
# Process a video file
python fall_detection_system.py --source video --input path/to/video.mp4 --output path/to/output.mp4 --draw-pose

# Process camera feed
python fall_detection_system.py --source camera --camera-id 0 --draw-pose
```

Command line arguments:

- `--source`: Choose between 'video' or 'camera' input
- `--input`: Path to input video file (when source is 'video')
- `--output`: Path to output video file (when source is 'video')
- `--camera-id`: Camera device ID (default: 0)
- `--draw-pose`: Flag to enable pose visualization

### Python API

#### Initialize the System

```python
from fall_detection_system import FallDetectionSystem

fall_detection_system = FallDetectionSystem()
```

#### Process a Single Frame

```python
# Process frame with optional pose visualization and detailed output
processed_frame, results = fall_detection_system.process_frame(
    frame,
    draw_pose=True,
    detailed_output=True
)
```

#### Process a Video File

```python
fall_detection_system.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    speed_factor=1,
    draw_pose=True
)
```

#### Process Camera Feed

```python
fall_detection_system.process_camera_feed(
    camera_id=0,
    draw_pose=True
)
```

## Output

- When processing video files, the system generates an annotated video with fall detection results
- For camera feed, results are displayed in real-time
- Green bounding boxes indicate normal activity
- Red bounding boxes indicate detected falls
- Optional pose keypoints and connections visualization
