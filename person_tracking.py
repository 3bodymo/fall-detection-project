from config import SEQUENCE_LENGTH

class Person:
    def __init__(self, bbox):
        """
        Initialize a Person instance with a bounding box.

        Args:
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2) defining the region of interest 
                         for the person.
        """
        self.bbox = bbox
        self.sequence = []
        self.fall_detected = False
        self.last_fall_time = 0
        self.prediction_prob = 0.0

    def add_pose(self, pose_features):
        """
        Add a sequence of pose features for the person.

        Args:
            pose_features (numpy.ndarray): A 1D array of pose features extracted for the current frame.
        """
        self.sequence.append(pose_features)
        # Keep only the last SEQUENCE_LENGTH frames
        if len(self.sequence) > SEQUENCE_LENGTH:
            self.sequence.pop(0)

    def is_ready_for_prediction(self):
        """
        Check if the person has enough pose sequences to make a prediction.

        Returns:
            bool: True if the person has a complete sequence of length `SEQUENCE_LENGTH`, otherwise False.
        """
        return len(self.sequence) == SEQUENCE_LENGTH

class PersonTracker:
    def __init__(self, iou_threshold=0.3, max_age=30):
        """
        Initialize the PersonTracker with specified IoU threshold and maximum age for tracking.

        Args:
            iou_threshold (float): Minimum IoU required to consider two bounding boxes as a match. Defaults to 0.3.
            max_age (int): Maximum age of a tracked person before they are removed. Defaults to 30.
        """
        self.persons = {}        # Dictionary to track persons with unique IDs
        self.next_id = 0         # Next available unique ID for a new person
        self.iou_threshold = iou_threshold  # Threshold for IoU
        self.max_age = max_age    # Maximum age for a tracked person

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1 (tuple): Coordinates of the first bounding box (x1, y1, x2, y2).
            bbox2 (tuple): Coordinates of the second bounding box (x3, y3, x4, y4).

        Returns:
            float: IoU value between the two bounding boxes.

        Steps:
        1. Calculate intersection coordinates.
        2. Compute intersection area.
        3. Calculate the areas of both bounding boxes.
        4. Compute the union area.
        5. Compute IoU as the intersection area divided by the union area.
        """
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        # Calculate intersection coordinates
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        # Check if boxes intersect
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)
        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area
        return iou

    def track_person(self, bbox):
        """
        Track a person by finding the best match with existing tracked persons.
        Assigns a unique ID to new or unmatched persons.

        Args:
            bbox (tuple): Coordinates of the bounding box (x1, y1, x2, y2) for the person.

        Returns:
            int: Unique ID assigned to the person.

        Steps:
        1. Clean up old tracks by removing persons older than the maximum age (`max_age`).
        2. Iterate through existing persons to find the best match based on IoU.
        3. If a match is found with IoU above the threshold, update the person's bounding box and reset their age.
        4. If no match is found, create a new track with a unique ID.
        5. Increment the age of all tracked persons.
        """
        best_match = None
        best_iou = 0

        # Clean up old tracks
        self.persons = {
            pid: person for pid, person in self.persons.items()
            if person['age'] < self.max_age
        }

        # Find best match based on IoU
        for pid, person in self.persons.items():
            iou = self.calculate_iou(bbox, person['bbox'])
            if iou > best_iou and iou > self.iou_threshold:
                best_match = pid
                best_iou = iou

        if best_match is not None:
            # Update existing track
            self.persons[best_match]['bbox'] = bbox
            self.persons[best_match]['age'] = 0
            return best_match
        else:
            # Create new track
            new_id = self.next_id
            self.persons[new_id] = {
                'bbox': bbox,
                'age': 0,
                'object': Person(bbox)
            }
            self.next_id += 1
            return new_id

    def increment_age(self):
        """
        Increment age for all tracked persons.

        This method is used to track how long a person has been tracked without any significant updates 
        (based on age).
        """
        for person in self.persons.values():
            person['age'] += 1