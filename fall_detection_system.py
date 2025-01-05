import cv2
from ultralytics import YOLO
from person_tracking import PersonTracker
from pose_processing import process_person
from inference import FallDetector
from utils import draw_overlay
from config import CONNECTIONS, LSTM_MODEL_PATH, YOLO_MODEL_PATH, CONF_THRESHOLD_YOLO, IOU_THRESHOLD_PT

class FallDetectionSystem:
    """
    A system for detecting falls in video frames using YOLO for person detection,
    a person tracker, and an LSTM-based fall detector.
    """
    def __init__(self):
        """Initialize the fall detection system with YOLO model, person tracker, and fall detector"""
        self.conf_threshold = CONF_THRESHOLD_YOLO
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.person_tracker = PersonTracker(iou_threshold=IOU_THRESHOLD_PT)
        self.fall_detector = FallDetector(LSTM_MODEL_PATH)

    def process_frame(self, frame, draw_pose=False, detailed_output=False):
        """
        Process a single frame and return the annotated frame and fall detection results
        
        Args:
            frame: Input frame
            draw_pose: Whether to draw pose keypoints and connections
            detailed_output: Whether to return detailed fall detection results
        
        Returns:
            tuple: If detailed_output=True:
                    (processed_frame, fall_detections)
                    - processed_frame: Frame with annotations
                    - fall_detections: List of tuples (person_id, is_fall, probability)
                  If detailed_output=False:
                    (processed_frame, has_fall)
                    - processed_frame: Frame with annotations
                    - has_fall: Boolean indicating if any fall was detected
        """
        raw_frame = frame.copy()
        fall_detections = []
        has_fall = False
        
        # Detect people using YOLO
        results = self.yolo_model(raw_frame, classes=0, conf=self.conf_threshold)
        self.person_tracker.increment_age()
        
        for result in results[0].boxes:
            bbox = tuple(map(int, result.xyxy[0]))
            person_id = self.person_tracker.track_person(bbox)
            person = self.person_tracker.persons[person_id]['object']
            
            pose_features = process_person(raw_frame, bbox)

            if pose_features is not None:
                person.add_pose(pose_features[0])
                if person.is_ready_for_prediction():
                    is_fall, probability = self.fall_detector.predict(person.sequence)
                    person.fall_detected = is_fall
                    person.prediction_prob = probability
                    if is_fall: has_fall = True
                    fall_detections.append((person_id, is_fall, probability))

            if draw_pose: draw_overlay(frame, bbox, person, CONNECTIONS, person_id, draw_pose=True)
            else: draw_overlay(frame, bbox, person, CONNECTIONS, person_id, draw_pose=False)

        return (frame, fall_detections) if detailed_output else (frame, has_fall)

    def process_video(self, video_path, output_path, speed_factor=1, draw_pose=False):
        """
        Process a video file and save the annotated video with fall detection results
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video file
            speed_factor: Factor to adjust the speed of the output video
            draw_pose: Whether to draw pose keypoints and connections
        
        Returns:
            None
        """
        # Open the video file and set up the video writer
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_fps = fps * speed_factor

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video file reached.")
                break

            frame_count += 1
            if frame_count % int(speed_factor) != 0:
                continue

            # Process each frame and write to the output video
            processed_frame, fall_detections = self.process_frame(frame, draw_pose)
            out.write(processed_frame)
            
            # Display the processed frame
            cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Fall Detection', processed_frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "test_videos/our_multi_test.mp4"
    output_path = "test_videos/multi_test_output_video.mp4"
    
    fall_detection_system = FallDetectionSystem()
    fall_detection_system.process_video(video_path, output_path)
