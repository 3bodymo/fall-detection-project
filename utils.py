import cv2
import numpy as np

def calculate_velocity(positions, time_step=1):
    """
    Calculate the velocity of landmarks based on their positions over time.

    Args:
        positions (numpy.ndarray): A 2D array of shape (frames × landmarks) representing 
                                    the positions of landmarks over time.
        time_step (float, optional): The time interval between consecutive frames. Defaults to 1.

    Returns:
        numpy.ndarray: A 2D array of the same shape as `positions` containing the calculated 
                       velocities for each landmark at each frame.
    """
    return np.diff(positions, axis=0, prepend=positions[0].reshape(1, -1)) / time_step

def calculate_acceleration(velocities, time_step=1):
    """
    Calculate the acceleration of landmarks based on their velocities over time.

    Args:
        velocities (numpy.ndarray): A 2D array of shape (frames × landmarks) representing 
                                     the velocities of landmarks over time.
        time_step (float, optional): The time interval between consecutive frames. Defaults to 1.

    Returns:
        numpy.ndarray: A 2D array of the same shape as `velocities` containing the calculated 
                       accelerations for each landmark at each frame.
    """
    return np.diff(velocities, axis=0, prepend=velocities[0].reshape(1, -1)) / time_step

def calculate_angle(v1, v2):
    """
    Calculate the angle (in radians) between two vectors.

    Args:
        v1 (numpy.ndarray): A 2D array where each row represents a vector.
        v2 (numpy.ndarray): A 2D array of the same shape as `v1` where each row represents a vector.

    Returns:
        numpy.ndarray: A 1D array of angles (in radians) for each pair of vectors in `v1` and `v2`.
    """
    dot_product = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

def extract_features(positions):
    """
    Extract motion and orientation features from pose landmarks.
    
    Args:
        positions (numpy.ndarray): A 2D array of shape (frames × landmarks × coordinates) representing 
                                    the pose landmarks. Each landmark has X and Y coordinates.
    
    Returns:
        numpy.ndarray: A 2D array where each row corresponds to a frame and contains the following features:
            - Original landmark positions (X, Y).
            - Velocities of landmarks.
            - Accelerations of landmarks.
            - Relative positions of landmarks with respect to the hip center.
            - Angle between the spine vector and the leg vector.
            - Change in height over time.
            - Body orientation angle relative to the vertical axis.
    """
    positions = positions.reshape(-1, 26)
    
    velocities = calculate_velocity(positions)
    accelerations = calculate_acceleration(velocities)
    
    hip_center = (positions[:, [6, 7]] + positions[:, [8, 9]]) / 2
    relative_positions = positions - np.tile(hip_center, 13)
    
    spine_vector = positions[:, [2, 3]] - positions[:, [6, 7]]
    leg_vector = positions[:, [10, 11]] - positions[:, [6, 7]]
    spine_leg_angle = calculate_angle(spine_vector, leg_vector)
    
    height = positions[:, 1]
    height_change = np.diff(height, prepend=height[0])
    
    vertical = np.array([0, 1])
    body_orientation = calculate_angle(spine_vector, np.tile(vertical, (len(spine_vector), 1)))
    
    features = np.concatenate([
        positions, velocities, accelerations, relative_positions,
        spine_leg_angle.reshape(-1, 1), height_change.reshape(-1, 1),
        body_orientation.reshape(-1, 1)
    ], axis=1)
    
    return features

def draw_overlay(frame, bbox, person, connections, person_id, draw_pose):
    """
    Draw bounding box, pose landmarks, pose connections, and fall detection status.
    
    Args:
        frame (numpy.ndarray): The current video frame (image) where the overlay will be drawn.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) specifying the top-left 
                      and bottom-right corners of the bounding box.
        person (object): An object containing the person's pose sequence, fall detection status, 
                         and prediction probability. Expected attributes:
                         - `fall_detected` (bool): Whether a fall was detected.
                         - `prediction_prob` (float): Probability of the fall prediction.
                         - `sequence` (list): List of pose landmark sequences, where each sequence 
                           contains normalized X, Y coordinates.
        connections (list of tuples): Pairs of indices specifying which landmarks to connect 
                                      with lines to visualize the pose.
        person_id (int): Unique identifier for the person being tracked.
    
    Returns:
        None: The function modifies the `frame` in place to draw the overlay.
    """
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if person.fall_detected else (0, 255, 0)
    label = f"ID: {person_id} - {'Fall' if person.fall_detected else 'No Fall'} ({person.prediction_prob:.2f})"
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if draw_pose:
        # Draw pose landmarks if available
        if person.sequence:  # Ensure there's a pose sequence
            latest_pose = np.array(person.sequence[-1][:26]).reshape(-1, 2)  # Extract pose landmarks
            for point in latest_pose:
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)  # Blue color (BGR)
        
        # Draw connections
        for start_idx, end_idx in connections:
            x1, y1 = int(latest_pose[start_idx][0] * frame.shape[1]), int(latest_pose[start_idx][1] * frame.shape[0])
            x2, y2 = int(latest_pose[end_idx][0] * frame.shape[1]), int(latest_pose[end_idx][1] * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # Blue color (BGR)