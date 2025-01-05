import os

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Landmarks indexes used for pose keypoints
LANDMARK_INDEXES = [0, 11, 12, 23, 24, 25, 26, 27, 28, 15, 16, 13, 14]

# Sequence length for LSTM model input
SEQUENCE_LENGTH = 10

# Connections between landmarks to visualize pose connections
CONNECTIONS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), 
               (6, 8), (1, 12), (12, 10), (2, 11), (11, 9)]

# Base directory - dynamically set to the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Debug output directory
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, 'debug')

# Path to the LSTM model for fall detection
LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fall_detection_lstm_model.keras')

# Path to the YOLO model for object detection
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11l.pt')

# Intersection over Union (IoU) threshold for object detection post-processing
IOU_THRESHOLD_PT = 0.1

# Confidence threshold for YOLO model predictions
CONF_THRESHOLD_YOLO = 0.9