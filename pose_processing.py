import numpy as np
import mediapipe as mp
from utils import extract_features
import cv2
import numpy as np
from config import LANDMARK_INDEXES
import uuid
from config import DEBUG_OUTPUT_DIR

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_person(frame, bbox):
    """
    Process a person: detect pose, extract features, and return them.
    Focuses on the bounding box region with an added margin to ensure 
    the region of interest is fully captured.
    
    Args:
        frame (numpy.ndarray): The input video frame (image) in BGR format.
        bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) specifying 
                      the region of interest for the person.

    Returns:
        numpy.ndarray or None: Extracted features for the LSTM model if pose landmarks are detected, 
                               otherwise None.

    Steps:
    1. Extract the dimensions of the frame for scaling and boundary checks.
    2. Add a margin around the bounding box to include some additional context.
    3. Ensure the bounding box with margin does not exceed frame boundaries.
    4. Crop the region of interest from the frame and convert it to RGB format for processing.
    5. Save the cropped image to a specified directory with a unique filename.
    6. Use the pose detection model (`pose.process`) to detect pose landmarks in the cropped frame.
    7. If landmarks are detected:
        a. Select only the relevant landmarks (specified in `LANDMARK_INDEXES`).
        b. Normalize the landmark coordinates to the frame dimensions for consistency.
        c. Flatten the landmark coordinates into a 1D array.
        d. Pass the normalized landmarks to the `extract_features` function to calculate motion and orientation features.
        e. Return the extracted features.
    8. If no landmarks are detected, log a message and return None.
    """
    height, width, _ = frame.shape  # Frame dimensions for normalization
    x1, y1, x2, y2 = bbox  # Bounding box coordinates

    # Add margin to the bounding box
    margin = 100
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)

    # Crop the bounding box area with margin and convert to RGB
    cropped_frame = frame[y1:y2, x1:x2]
    cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

    # Generate a random unique filename
    unique_filename = f"{uuid.uuid4().hex}.png"
    image_path = f"{DEBUG_OUTPUT_DIR}/{unique_filename}"
    cv2.imwrite(image_path, cv2.cvtColor(cropped_frame_rgb, cv2.COLOR_RGB2BGR))

    # Process the cropped region
    person_results = pose.process(cropped_frame_rgb)

    if person_results.pose_landmarks:
        # Select relevant landmarks
        selected_landmarks = [
            person_results.pose_landmarks.landmark[i] for i in LANDMARK_INDEXES
        ]

        # Normalize landmarks relative to the bounding box dimensions
        pose_landmarks = [
            [(lmk.x * (x2 - x1) + x1) / width,  # Scale back to frame coordinates
             (lmk.y * (y2 - y1) + y1) / height]
            for lmk in selected_landmarks
        ]
        pose_landmarks_flat = np.array(pose_landmarks).flatten()

        # Extract features for the LSTM model
        features = extract_features(pose_landmarks_flat.reshape(1, -1))
        return features
    else:
        print(f"No pose landmarks detected for bounding box: {bbox}")
        return None


