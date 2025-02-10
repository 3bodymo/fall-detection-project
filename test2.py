import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import http.client, urllib.parse
import time

from ultralytics import YOLO
yolo_model = YOLO('yolo11n.pt') 
import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.video import Video


# Load the new trained LSTM model
model = load_model('E:/Users/Настя/Downloads/thws/Project/app/latest/fall_detection_lstm_model2.keras')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the indexes of landmarks to track (all 13 landmarks)
landmark_indexes = [0, 11, 12, 23, 24, 25, 26, 27, 28, 15, 16, 13, 14]

# Define connections between landmarks
connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (1, 12), (12, 10), (2, 11), (11, 9)]

# Path to the local MP4 video file
video_path = "E:/Users/Настя/Downloads/thws/Project/app/latest/test2.mp4"
# video_path = "rtsp://admin:Fall_Detection0@192.168.0.100:554/h264Preview_01_sub"

output_path = "E:/Users/Настя/Downloads/thws/Project/app/latest/output_video.mp4"  # Output video file
speed_factor = 1  # Default speed factor

# Open the video file and get its properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_fps = fps * speed_factor  # Adjust frame rate based on speed

# Initialize the VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
# out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))

# Input pushover User and API key below
user_key = ""
api_token = ""

# Initialize variables for LSTM input
sequence_length = 10
pose_sequence = []
fall_detected = False
last_fall_time = 0
fall_segment_writer = None
fall_segment_path = "E:/Users/Настя/Downloads/thws/Project/app/latest/fall_segment_{}.mp4"
fall_sequence = []
fall_sequence_threshold = 2
fall_recording = False
non_fall_frame_count = 0
fall_frame_count = 0
fall_segment_index = 0
fall_segments_folder = "E:/Users/Настя/Downloads/thws/Project/app/latest/"
fall_segment_template = "fall_segment_{}.mp4"

# New feature extraction functions
def calculate_velocity(positions, time_step=1):
    return np.diff(positions, axis=0, prepend=positions[0].reshape(1, -1)) / time_step

def calculate_acceleration(velocities, time_step=1):
    return np.diff(velocities, axis=0, prepend=velocities[0].reshape(1, -1)) / time_step

def calculate_angle(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

def extract_features(positions):
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
        positions,
        velocities,
        accelerations,
        relative_positions,
        spine_leg_angle.reshape(-1, 1),
        height_change.reshape(-1, 1),
        body_orientation.reshape(-1, 1)
    ], axis=1)
    
    return features

#===================================================================================


def update(frame):
    # global image
    # Проверяем, что кадр не пустой
    if frame is not None:
        # Преобразуем кадр в текстуру Kivy
        buffer = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        # Обновляем текстуру виджета
        image.texture = texture



def toggle_view():
    if is_video_shown:
        # Переход на экран с другим содержимым
        video_layout.clear_widgets()
        video_layout.add_widget(Button(text="Here will be alarm records", size_hint=(1, 1)))
        button.text = "Show video stream"
    else:
        # Возврат к видеопотоку
        video_layout.clear_widgets()
        video_layout.add_widget(image)
        button.text = "Show alarm records"
    is_video_shown = not is_video_shown



# Основной контейнер
main_container = BoxLayout(orientation='vertical', padding=20, spacing=10)

# Контейнер для видео
video_layout = BoxLayout(size_hint=(1, 0.8))
    # Виджет для отображения видео
image = Image(size_hint=(1, 1))  # Видео адаптируется по размеру контейнера
video_layout.add_widget(image)

# Состояние отображения (видеопоток или другое содержимое)
# state['is_video_shown'] = True

# Контейнер для кнопки
button_layout = BoxLayout(size_hint=(1, 0.2), padding=[20, 10])
button = Button(text="Show alarm records", size_hint=(None, None), size=(300, 50))
# Передаём словарь состояния:
button.bind(on_press=lambda instance: toggle_view)
button_layout.add_widget(Widget())  # Пустое пространство слева
button_layout.add_widget(button)
button_layout.add_widget(Widget())  # Пустое пространство справа

# Добавляем видео и кнопку в основной контейнер
main_container.add_widget(video_layout)
main_container.add_widget(button_layout)
is_video_shown = True  # Используем словарь для хранения состояния


# def update_kivy(frame):
#     ret, frame = cap.read()
#     if ret:
#         update(frame)
        
    # Обновление кадра каждую итерацию
# Clock.schedule_interval(update, 1.0 / 30.0)



#===================================================================================




# Основной цикл обработки видео
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        break

    frame_count += 1
    if frame_count % int(speed_factor) != 0:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        selected_pose_landmarks = [results.pose_landmarks.landmark[i] for i in landmark_indexes]
        pose_landmarks = [[lmk.x, lmk.y] for lmk in selected_pose_landmarks]
        pose_landmarks_flat = np.array(pose_landmarks).flatten()

        current_features = extract_features(pose_landmarks_flat.reshape(1, -1))
        pose_sequence.append(current_features[0])
        if len(pose_sequence) > sequence_length:
            pose_sequence.pop(0)

        if len(pose_sequence) == sequence_length:
            lstm_input = np.array([pose_sequence])
            prediction = model.predict(lstm_input)

            if prediction[0][0] > 0.5:
                skeleton_color = (0, 255, 0)
                text = 'No Fall'
                fall_frame_count = 0
                non_fall_frame_count += 1
                if non_fall_frame_count >= fall_sequence_threshold and fall_recording:
                    # Заканчиваем запись фрагмента падения
                    fall_recording = False
                    fall_segment_writer.release()
                    fall_segment_writer = None
            else:
                skeleton_color = (0, 0, 255)
                text = 'Fall'
                non_fall_frame_count = 0
                fall_frame_count += 1
                if fall_frame_count >= fall_sequence_threshold and not fall_recording:
                    # Начинаем запись нового фрагмента
                    fall_recording = True
                    fall_segment_index += 1
                    fall_segment_writer = cv2.VideoWriter(
                        fall_segment_path.format(fall_segment_index),
                        fourcc, new_fps, (frame_width, frame_height)
                    )

            cv2.putText(frame, text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, skeleton_color, 2, cv2.LINE_AA)
            for landmark in selected_pose_landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, skeleton_color, -1)

            for connection in connections:
                start_idx, end_idx = connection
                start_landmark = selected_pose_landmarks[start_idx]
                end_landmark = selected_pose_landmarks[end_idx]
                start_x, start_y = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                end_x, end_y = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                cv2.line(frame, (start_x, start_y), (end_x, end_y), skeleton_color, 2)
                
    # Сохраняем кадр в основной выходной файл
    # out.write(frame)
    
    # Если записывается сегмент падения, сохраняем его
    if fall_recording and fall_segment_writer:
        fall_segment_writer.write(frame)


    update(frame)
    cv2.imshow('Fall Detection', frame) 
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Завершаем запись  
if fall_segment_writer:
    fall_segment_writer.release()
cap.release()
# out.release()
cv2.destroyAllWindows()




    
    
