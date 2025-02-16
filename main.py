# TESTING PART======================================================
# self.video_path = "E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/test2.mp4"
# self.cap = cv2.VideoCapture(0)  # '0' - the first camera (web camera)
# video_path = "rtsp://admin:Fall_Detection0@192.168.0.100:554/h264Preview_01_sub"

# telegram_token = 7856607431:AAHuYzNHSJh04soAf_2Fks46WAFhhDXIlwU
# =================================================================

import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import http.client, urllib.parse
import time
import threading
from ultralytics import YOLO
yolo_model = YOLO('yolo11n.pt') 
import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.video import Video
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
import requests
import json
from playsound import playsound
from kivy.graphics import Color, Line, Rectangle
from kivy.core.window import Window
import pygame


class VideoApp(App):
    def build(self):
        self.video_path =""
        self.bot_token = ""
        self.chat_id = None
        # Load the new trained LSTM model
        self.model = load_model('E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/fall_detection_lstm_model2.keras')

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Define the indexes of landmarks to track (all 13 landmarks)
        self.landmark_indexes = [0, 11, 12, 23, 24, 25, 26, 27, 28, 15, 16, 13, 14]

        # Define connections between landmarks
        self.connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (1, 12), (12, 10), (2, 11), (11, 9)]

        # Path to the local MP4 video file


        self.output_path = "E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/output_video.mp4"  # Output video file
        self.speed_factor = 1  # Default speed factor

        # Open the video file and get its properties
        self.cap = cv2.VideoCapture(self.video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_fps = 0.08 * self.fps * self.speed_factor  # Adjust frame rate based on speed

        # Initialize the VideoWriter to save the output
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        # out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))

        # Input pushover User and API key below
        self.cap = None

        # Fall detection & recording related variables
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        self.new_fps = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.user_key = ""
        self.api_token = ""
        self.sequence_length = 10
        self.pose_sequence = []
        self.fall_detected = False
        self.fall_segment_writer = None
        self.fall_segment_path = "E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/records/fall_segment_{}.mp4"
        self.fall_sequence = []
        self.fall_sequence_threshold = 2
        self.fall_recording = False
        self.non_fall_frame_count = 0
        self.fall_frame_count = 0
        self.fall_segment_index = 0
        self.fall_segments_folder = "E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/records"
        self.fall_segment_template = "fall_segment_{}.mp4"
        self.frame_count = 0
        self.thread = None
        self.video_list = []
        self.current_video = None
        self.alarm_playing = False
        self.alarm_sound_path = "E:/Users/Настя/Downloads/thws/Project/app/fall-detection-project/alert.wav"
        
 
        Window.clearcolor = (0.85, 0.85, 0.85, 1)
        # The main container
        self.main_container = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Video container
        self.video_layout = BoxLayout(size_hint=(1, 1))
        # Widget for displaying video
        self.image = Image(size_hint=(1, 1))  # The video adapts to the container size


        self.video_layout.add_widget(self.image)

        # Container for button
        button_layout = BoxLayout(size_hint=(1, 0.2), padding=[20, 10])
        # self.button = Button(text="Show alarm records", size_hint=(None, None), size=(300, 50))
        
        
        self.button = Button(
            text="Show alarm records",
            size_hint=(None, None),  
            size=(300, 50),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            background_normal="",  # Disable default texture
            color=(0.3, 0.15, 0.05, 1),  # Dark brown text
            background_color=(1, 1, 0.8, 1)
        )
        # Custom styling using canvas.before
        with self.button.canvas.before:
            # Border (2px Dark Brown)
            Color(0.3, 0.15, 0.05, 1)  # Dark Brown RGBA
            border_line = Line(width=2, points=[
                self.button.x, self.button.y,  # Bottom-left
                self.button.right, self.button.y,  # Bottom-right
                self.button.right, self.button.top,  # Top-right
                self.button.x, self.button.top,  # Top-left
                self.button.x, self.button.y  # Closing the rectangle
            ])

        # Assign the drawn objects to the button so they can be updated
        self.button.border_line = border_line

        # Ensure updates when resized/moved
        self.button.bind(pos=self.update_bg, size=self.update_bg)
        
        self.button.bind(on_press=self.toggle_view)
        button_layout.add_widget(Widget())  # Empty space on the left
        button_layout.add_widget(self.button)
        button_layout.add_widget(Widget())  # Empty space on the right

        # Adding a video and a button to the main container
        self.main_container = BoxLayout(orientation='vertical', padding=20, spacing=10)
    # (setup self.video_layout, self.image, button_layout, etc.)
        self.main_container.add_widget(self.video_layout)
        self.main_container.add_widget(button_layout)
        
        self.is_video_shown = True
        self.running = True
        self.video_thread = threading.Thread(target=self.main_loop, daemon=True)
        
        # Now create a root container that will initially display the camera URL interface.
        self.root_container = BoxLayout()
        # Build and add the camera URL interface
        camera_url_interface = self.build_camera_url_interface()
        self.root_container.add_widget(camera_url_interface)
        
        # Instead of returning self.main_container, return the root container:
        return self.root_container
        
        # self.main_loop()
        # return self.main_container
    def update_bg(self, instance, *args):
        """Update the background and border when the button is resized or moved"""
        instance.border_line.points = [
            instance.x, instance.y,  # Bottom-left
            instance.right, instance.y,  # Bottom-right
            instance.right, instance.top,  # Top-right
            instance.x, instance.top,  # Top-left
            instance.x, instance.y  # Closing the rectangle
        ]
        
    def build_camera_url_interface(self):
        """Builds an interface that asks the user to input the camera URL."""
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        # label = Label(text="Enter camera URL", size_hint=(0.8, 0.06),  pos_hint={"center_x": 0.5, "center_y": 0.5})
        label = Label(
            text="Enter camera URL",
            size_hint=(0.8, 0.06),
            # height=30,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            color=(0.3, 0.15, 0.05, 1)  # Dark brown text
        )
        layout.add_widget(label)
        self.url_input = TextInput(
            text="",
            multiline=False,
            size_hint=(0.8, 0.06),
            # height=50,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            foreground_color=(0.3, 0.15, 0.05, 1),  # Dark brown text
            background_color=(1, 1, 1, 1)  # White background for input
        )
        layout.add_widget(self.url_input)
        
        label_token = Label(
            text="Enter Telegram Bot Token",
            size_hint=(0.8, 0.06),
            # height=30,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            color=(0.3, 0.15, 0.05, 1)  # Dark brown text
        )
        layout.add_widget(label_token)
        self.token_input = TextInput(
            text="7856607431:AAHuYzNHSJh04soAf_2Fks46WAFhhDXIlwU",
            multiline=False,
            size_hint=(0.8, 0.06),
            # height=50,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            foreground_color=(0.3, 0.15, 0.05, 1),  # Dark brown text
            background_color=(1, 1, 1, 1)  # White background for input
        )
        layout.add_widget(self.token_input)
        
        # ok_button = Button(text="OK", size_hint=(0.3, 0.03), pos_hint={"center_x": 0.5, "center_y": 0.5})
        
        ok_button = Button(
            text="OK",
            size_hint=(None, None),  # Smaller button
            size=(300, 50),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            background_normal="",  # Disable default texture
            color=(0.3, 0.15, 0.05, 1),  # Dark brown text
            background_color=(1, 1, 0.8, 1)
        )
        # Custom styling using canvas.before
        with ok_button.canvas.before:
            # Border (2px Dark Brown)
            Color(0.3, 0.15, 0.05, 1)  # Dark Brown RGBA
            border_line = Line(width=2, points=[
                ok_button.x, ok_button.y,  # Bottom-left
                ok_button.right, ok_button.y,  # Bottom-right
                ok_button.right, ok_button.top,  # Top-right
                ok_button.x, ok_button.top,  # Top-left
                ok_button.x, ok_button.y  # Closing the rectangle
            ])

        # Assign the drawn objects to the button so they can be updated
        ok_button.border_line = border_line

        # Ensure updates when resized/moved
        ok_button.bind(pos=self.update_bg, size=self.update_bg)
    
        ok_button.bind(on_press=self.on_camera_url_ok)
        layout.add_widget(ok_button)
        
        return layout

    
    
    
    def on_camera_url_ok(self, instance):
        """Handles the OK button press on the camera URL interface."""
        self.video_path = self.url_input.text.strip()
        self.bot_token = self.token_input.text.strip()
        
        if not self.video_path:
          print("Error: No camera URL provided.")
          return
        
        if not self.bot_token:
          print("Error: No number provided.")
          return

        if self.video_path.isdigit():
          self.video_path = int(self.video_path)  # Convert to integer for webcam index
        else:
          self.video_path = self.video_path

        self.chat_id = self.get_chat_id()
        
        if not self.chat_id:
          print("❌ Chat ID not found. Make sure to send '/start' to the bot.")
          return
      
        # Now initialize the video capture with the given URL
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Failed to open video source")
            return

        # Retrieve video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_fps = 0.08 * self.fps * self.speed_factor

        # Replace the camera URL interface with your main interface.
        self.root_container.clear_widgets()
        self.root_container.add_widget(self.main_container)
        
        # Start the video thread if it is not already running.
        if self.video_thread is None or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self.main_loop, daemon=True)
            self.video_thread.start()

    
    def get_chat_id(self):
      """Fetches chat_id dynamically when the bot receives a /start message."""
      url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
      response = requests.get(url)

      if response.status_code == 200:
          data = response.json()
          print("Response from Telegram API:", json.dumps(data, indent=4))  # Debugging

          for update in data.get("result", []):
              if "message" in update and "chat" in update["message"]:
                  self.chat_id = update["message"]["chat"]["id"]
                  print(f"✅ Chat ID found: {self.chat_id}")
                  return self.chat_id

      print("❌ Error: No chat_id found. Please send '/start' to the bot first.")
      return None

        
    def send_telegram_message(self, text):
      """Sends an alarm message to Telegram and logs the response."""
      if not self.chat_id:
          print("❌ Error: No chat_id found. Cannot send message.")
          return

      url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
      data = {"chat_id": self.chat_id, "text": text}

      response = requests.post(url, data=data)
      if response.status_code == 200:
          print("✅ Message sent successfully!")
      else:
          print("❌ Failed to send message. Response:", response.text)


    def send_telegram_video(self, video_path):
      """Sends a recorded fall video to Telegram."""
      if not self.chat_id:
          print("❌ Error: No chat_id found. Cannot send video.")
          return

      if not os.path.exists(video_path):
          print(f"❌ Error: Video file '{video_path}' not found.")
          return

      url = f"https://api.telegram.org/bot{self.bot_token}/sendVideo"
      with open(video_path, "rb") as video_file:
          response = requests.post(url, data={"chat_id": self.chat_id}, files={"video": video_file})

      if response.status_code == 200:
          print("✅ Video sent successfully!")
      else:
          print("❌ Failed to send video. Response:", response.text)

    
       
    def on_stop(self):
        # Stop the thread when the application closes
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        
  
    pygame.mixer.init()

    def play_alarm(self):
        """Play the alarm sound asynchronously (non-blocking)"""
        try:
            pygame.mixer.music.load(self.alarm_sound_path)  # Load the sound
            pygame.mixer.music.play()  # Play the sound without blocking
        except Exception as e:
            print(f"Error playing alarm: {e}")

    def trigger_alarm(self):
        """Trigger the alarm without blocking"""
        self.alarm_thread = threading.Thread(target=self.play_alarm)
        self.alarm_thread.daemon = True
        self.alarm_thread.start()
        
# ============Video_records part=================================================================================
    def video_records_container(self):
        video_container = BoxLayout(orientation='vertical', padding=20, spacing=10)
        # Container for displaying video
        video_image = Image(size_hint=(1, 0.8))
        video_container.add_widget(video_image)

        # Container for control buttons
        control_buttons = BoxLayout(size_hint=(1, 0.2), padding=10, spacing=10)
        back_button = Button(text="Back to Video List", size_hint=(1, None), height=50)
        control_buttons.add_widget(back_button)
        video_container.add_widget(control_buttons)
        
        control_buttons = BoxLayout(size_hint=(1, 0.2), padding=10, spacing=10)
        back_button = Button(text="Back to Video List", size_hint=(1, None), height=50)
        back_button.bind(on_press=lambda instance: self.show_video_list(video_container, control_buttons, video_image))
        control_buttons.add_widget(back_button)
        video_container.add_widget(control_buttons)

        # Initially show a list of videos
        self.show_video_list(video_container, control_buttons, video_image)

        return video_container
        
    def create_video_list_container(self, video_container, video_image, control_buttons):
        """Creates a container for displaying a list of videos.."""
        # Scrollable widget
        scroll_view = ScrollView(size_hint=(1, 1))
        grid = GridLayout(cols=1, spacing=10, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))

        # Updating the video list
        self.update_video_list()

        # Add buttons for each video
        for video_name in self.video_list:
            button = Button(text=video_name, size_hint=(1, None), height=50)
            button.id = video_name  # Set the button id equal to the video name
            button.bind(on_press=lambda instance: self.on_video_button_press(video_container, video_image, control_buttons, instance))
            grid.add_widget(button)
        scroll_view.add_widget(grid)
        return scroll_view

    def on_video_button_press(self, video_container, video_image, control_buttons, instance):
        """Processes a click on a button with a video."""
        selected_video_name = instance.id  # Get the video name from the button id
        print(f"Selected video: {selected_video_name}")
        # Start playing the video
        self.play_video(selected_video_name, video_container, video_image, control_buttons)
        
    def update_video_list(self):
        """Updating the list of videos in the folder."""
        self.video_list = [f for f in os.listdir(self.fall_segments_folder) if f.endswith('.mp4')]

    def show_video_list(self, video_container, control_buttons, video_image, instance=None):
        """Displays a list of available videos.."""
        video_container.clear_widgets()

        # Create a container for a video list
        video_list_container = self.create_video_list_container(video_container, video_image, control_buttons)

        # Adding a container to the main widget
        video_container.add_widget(video_list_container)


    def play_video(self, video_name, video_container, video_image, control_buttons):
        """Starts playing the selected video."""
        # clear the container
        video_container.clear_widgets()
        current_video_path = os.path.join(self.fall_segments_folder, video_name)
        self.capture = cv2.VideoCapture(current_video_path)

        # Add video and control widgets
        video_container.add_widget(video_image)
        video_container.add_widget(control_buttons)

        # Using Lambda for delayed call with arguments
        Clock.schedule_interval(
            lambda dt: self.update_frame(video_container, control_buttons, video_image),
            1.0 / self.new_fps
        )
# update frames for video records
    def update_frame(self, video_container, control_buttons, video_image): 
        """Обновляет кадр текущего видео."""
        if hasattr(self, 'capture') and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                frame = frame[::-1]  # rotate frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buf = frame.tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                video_image.texture = texture
            else:
                self.capture.release()
                Clock.unschedule(self.update_frame)  # Stop updating frames
                self.show_video_list(video_container, control_buttons, video_image)
#=====================================================================================================

    def update(self, frame):
        if frame is not None:
        # Convert the frame to a Kivy texture
            frame = frame[::-1]  # rotate frame
            buffer = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
            self.image.texture = texture
            

    def toggle_view(self, instance):
        # Remove content from video_layout
        self.video_layout.clear_widgets()

        if self.is_video_shown:
            # Switching to a container with records
            video_container = self.video_records_container()
            self.video_layout.add_widget(video_container)
            self.button.text = "Show video stream"
        else:
            # make sure that image is not attached to another parent
            if self.image.parent:
                self.image.parent.remove_widget(self.image)
            # add image
            self.video_layout.add_widget(self.image)
            self.button.text = "Show alarm records"

        # Switch the state
        self.is_video_shown = not self.is_video_shown


    # New feature extraction functions
    def calculate_velocity(self, positions, time_step=1):
        return np.diff(positions, axis=0, prepend=positions[0].reshape(1, -1)) / time_step

    def calculate_acceleration(self, velocities, time_step=1):
        return np.diff(velocities, axis=0, prepend=velocities[0].reshape(1, -1)) / time_step

    def calculate_angle(self, v1, v2):
        dot_product = np.sum(v1 * v2, axis=1)
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        return np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

    def extract_features(self, positions):
        positions = positions.reshape(-1, 26)
        
        velocities = self.calculate_velocity(positions)
        accelerations = self.calculate_acceleration(velocities)
        
        hip_center = (positions[:, [6, 7]] + positions[:, [8, 9]]) / 2
        relative_positions = positions - np.tile(hip_center, 13)
        
        spine_vector = positions[:, [2, 3]] - positions[:, [6, 7]]
        leg_vector = positions[:, [10, 11]] - positions[:, [6, 7]]
        spine_leg_angle = self.calculate_angle(spine_vector, leg_vector)
        
        height = positions[:, 1]
        height_change = np.diff(height, prepend=height[0])
        
        vertical = np.array([0, 1])
        body_orientation = self.calculate_angle(spine_vector, np.tile(vertical, (len(spine_vector), 1)))
        
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


    def main_loop(self):
    # the main video processing loop
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video file reached.")
                break

            self.frame_count += 1
            if self.frame_count % int(self.speed_factor) != 0:
                continue

            # Frame processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            self.results = self.pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Frame logic processing
            self.process_frame(frame)
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the widget with a new frame
            Clock.schedule_once(lambda dt: self.update(display_frame))

            # Key completion
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break

     
        
        
    def process_frame(self, frame):
        """Processes frame and manages fall recording with two stop conditions:
        1. Person gets up (no_fall)
        2. Person has been falling for 50 frames
        """
        if self.results.pose_landmarks:
            selected_pose_landmarks = [self.results.pose_landmarks.landmark[i] for i in self.landmark_indexes]
            pose_landmarks = [[lmk.x, lmk.y] for lmk in selected_pose_landmarks]
            pose_landmarks_flat = np.array(pose_landmarks).flatten()

            current_features = self.extract_features(pose_landmarks_flat.reshape(1, -1))
            self.pose_sequence.append(current_features[0])
            if len(self.pose_sequence) > self.sequence_length:
                self.pose_sequence.pop(0)

            if len(self.pose_sequence) == self.sequence_length:
                self.lstm_input = np.array([self.pose_sequence])
                prediction = self.model.predict(self.lstm_input)

                if prediction[0][0] > 0.5:
                    # "No Fall" - end recording if active
                    skeleton_color = (0, 255, 0)
                    text = 'No Fall'
                    self.fall_frame_count = 0
                    self.non_fall_frame_count += 1
                    
                    # End recording if person gets up
                    if self.fall_recording and self.non_fall_frame_count >= self.fall_sequence_threshold:
                        self.save_and_reset_recording()
                else:
                    # "Fall" detected
                    skeleton_color = (0, 0, 255)
                    text = 'Fall'
                    self.non_fall_frame_count = 0
                    self.fall_frame_count += 1

                    # Start recording if not already recording
                    if not self.fall_recording and self.fall_frame_count >= self.fall_sequence_threshold:
                        self.start_new_recording()
                    
                    if self.fall_recording and self.fall_frame_count == self.fall_sequence_threshold+3:
                        self.trigger_alarm()
                        self.send_telegram_message("🚨 Fall detected! Sending video soon.")
                         # Check if falling has continued for 20 frames
                    if self.fall_frame_count == 20:
                        self.trigger_alarm()
                        self.alarm_playing = False
                        self.save_and_reset_recording()
                        

                # Display text and skeleton on frame
                cv2.putText(frame, text, (frame.shape[1] - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, skeleton_color, 2, cv2.LINE_AA)
                
                # Draw landmarks
                for landmark in selected_pose_landmarks:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, skeleton_color, -1)

                # Draw connections
                for connection in self.connections:
                    start_idx, end_idx = connection
                    start_landmark = selected_pose_landmarks[start_idx]
                    end_landmark = selected_pose_landmarks[end_idx]
                    start_x, start_y = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                    end_x, end_y = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), skeleton_color, 2)

        # Write frame if recording is active
        if self.fall_recording and self.fall_segment_writer:
            self.fall_segment_writer.write(frame)

    def start_new_recording(self):
        """Starts a new fall recording segment."""
        self.fall_recording = True
        self.fall_segment_index += 1
        self.fall_segment_writer = cv2.VideoWriter(
            self.fall_segment_path.format(self.fall_segment_index),
            self.fourcc, self.new_fps, (self.frame_width, self.frame_height)
        )

    def save_and_reset_recording(self):
      """Saves the current recording and sends an alert message."""
      self.fall_recording = False
      if self.fall_segment_writer:
          self.fall_segment_writer.release()
          self.fall_segment_writer = None

      # Get the last recorded file path
      segment_video_path = self.fall_segment_path.format(self.fall_segment_index)

      # Send alert message with video
      if os.path.exists(segment_video_path):
        self.send_telegram_message("🚨 Fall detected! Sending video now.")
        self.send_telegram_video(segment_video_path)

       


app = VideoApp()
app.run()