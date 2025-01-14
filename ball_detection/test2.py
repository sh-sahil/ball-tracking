import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import deque
from typing import List, Union, Tuple
from scipy.signal import savgol_filter
from tqdm import tqdm
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
from moviepy.editor import VideoFileClip
import time
from tqdm import tqdm 

# Basic tracking parameters
BALL_ID = 0
MAXLEN = 15
DISTANCE_THRESHOLD = 80.0
CONFIDENCE_THRESHOLD = 0.3
MAX_FRAMES_WITHOUT_DETECTION = 8
POSITION_HISTORY_SIZE = 6
SMOOTHING_FACTOR = 0.6
RECENT_POSITIONS_LENGTH = 8

class VideoProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ball Tracker")
        self.root.geometry("660x550")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        self.create_widgets(scrollable_frame)
    
    def create_widgets(self, parent):
        row = 0
        
        # Model selection
        ttk.Label(parent, text="Model Path:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.model_path_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.model_path_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(parent, text="Select Model", command=self.select_model).grid(row=row, column=2, padx=5, pady=5)
        row += 1
        
        # Video selection
        ttk.Label(parent, text="Input Video:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.video_path_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.video_path_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(parent, text="Select Video", command=self.select_video).grid(row=row, column=2, padx=5, pady=5)
        row += 1
        
        # JSON config
        ttk.Label(parent, text="Config JSON:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.json_path_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.json_path_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(parent, text="Select JSON", command=self.select_json).grid(row=row, column=2, padx=5, pady=5)
        row += 1
        
        # Output name
        ttk.Label(parent, text="Output Name:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.output_name_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.output_name_var, width=50).grid(row=row, column=1, padx=5, pady=5)
        ttk.Label(parent, text=".mp4").grid(row=row, column=2, padx=5, pady=5, sticky="w")
        row += 1
        
        # Trail parameters
        trail_frame = ttk.LabelFrame(parent, text="Trail Parameters")
        trail_frame.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        self.trail_params = {
            "TRAIL_LENGTH": (tk.StringVar(value="30"), "Number of previous positions to use"),
            "TRAIL_WIDTH": (tk.StringVar(value="3"), "Width of the trail in pixels"),
            "TRAIL_SPEED": (tk.StringVar(value="0.5"), "Speed factor for trail fade"),
            "TRAIL_COLOR": (tk.StringVar(value="(255,255,0)"), "Color in BGR format"),
            "EMA_ALPHA": (tk.StringVar(value="0.3"), "EMA smoothing factor (0-1)")
        }
        
        for i, (param, (var, comment)) in enumerate(self.trail_params.items()):
            ttk.Label(trail_frame, text=param + ":").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            ttk.Entry(trail_frame, textvariable=var, width=15).grid(row=i, column=1, padx=5, pady=2)
            ttk.Label(trail_frame, text=comment).grid(row=i, column=2, padx=5, pady=2, sticky="w")
        
        row += 1
        
        # Progress bar and status
        progress_frame = ttk.LabelFrame(parent, text="Progress")
        progress_frame.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(pady=5)
        
        row += 1
        
        # Start button
        self.start_button = ttk.Button(parent, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=row, column=0, columnspan=3, pady=20)
    
    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt")])
        if path:
            self.model_path_var.set(path)
    
    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi"),
            ("MP4 files", "*.mp4"),
            ("MKV files", "*.mkv"),
            ("AVI files", "*.avi")
        ])
        if path:
            self.video_path_var.set(path)
    
    def select_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            self.json_path_var.set(path)
            self.load_json_config(path)
    
    def load_json_config(self, path):
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                if 'triangle' in config:
                    # Update triangle parameters if needed
                    pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {str(e)}")
    
    def start_processing(self):
        if not all([
            self.model_path_var.get(),
            self.video_path_var.get(),
            self.output_name_var.get()
        ]):
            messagebox.showerror("Error", "Please fill in all required fields")
            return
        
        self.start_button.config(state="disabled")
        self.status_var.set("Starting processing...")
        
        # Create output path
        output_path = f"{self.output_name_var.get()}.mp4"
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_video, args=(output_path,), daemon=True).start()
    
    def process_video(self, output_path):
        try:
            tracker = BallTracker(
                model_path=self.model_path_var.get(),
                source_video=self.video_path_var.get(),
                output_path=output_path,
                config_json=self.json_path_var.get() if self.json_path_var.get() else None,
                trail_length=int(self.trail_params["TRAIL_LENGTH"][0].get()),
                trail_width=int(self.trail_params["TRAIL_WIDTH"][0].get()),
                trail_speed=float(self.trail_params["TRAIL_SPEED"][0].get()),
                trail_color=self.trail_params["TRAIL_COLOR"][0].get(),
                ema_alpha=float(self.trail_params["EMA_ALPHA"][0].get())
            )
            # Set the progress callback
            tracker.set_progress_callback(self.update_progress)
            
            # Process video
            tracker.process_video()
            
            # Add audio
            self.status_var.set("Adding audio...")
            self.add_audio(output_path)
            
            self.status_var.set("Processing complete!")
            messagebox.showinfo("Success", "Video processing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.start_button.config(state="normal")
    
    def update_progress(self, current_frame, total_frames, remaining_time):
        if total_frames > 0:
            progress = (current_frame / total_frames) * 100
            self.progress_var.set(progress)
            if remaining_time is not None:
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                self.status_var.set(f"Processing: {progress:.1f}% - Remaining Time: {formatted_time}")
            else:
                self.status_var.set(f"Processing: {progress:.1f}%")
        else:
            self.status_var.set("Processing...")
        self.root.update()
    
    def add_audio(self, output_path):
        try:
            # Load videos
            original = VideoFileClip(self.video_path_var.get())
            processed = VideoFileClip(output_path)
            
            # Add audio
            final = processed.set_audio(original.audio)
            
            # Save with audio
            temp_output = f"temp_{output_path}"
            final.write_videofile(temp_output)
            
            # Cleanup
            original.close()
            processed.close()
            final.close()
            
            # Replace original with audio version
            os.replace(temp_output, output_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add audio: {str(e)}")
    
    def run(self):
        self.root.mainloop()

def replace_outliers_based_on_distance(
    positions: List[np.ndarray],
    distance_threshold: float
) -> List[np.ndarray]:
    last_valid_position: Union[np.ndarray, None] = None
    cleaned_positions: List[np.ndarray] = []
    for position in positions:
        if len(position) == 0:
            cleaned_positions.append(position)
        else:
            if last_valid_position is None:
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                distance = np.linalg.norm(position - last_valid_position)
                if distance > distance_threshold:
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    cleaned_positions.append(position)
                    last_valid_position = position

    return cleaned_positions


class BallTracker:
    def __init__(self, model_path: str, source_video: str, output_path: str,
                 config_json: str = None,
                 trail_length: int = 30,
                 trail_width: int = 3,
                 trail_speed: float = 0.5,
                 trail_color: tuple = (255, 255, 0),
                 ema_alpha: float = 0.3):
        # Initialization code remains the same...
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Video file not found: {source_video}")
        
        self.config = self.load_config(config_json) if config_json else {}
        # Set triangle parameters from config
        triangle_config = self.config.get('triangle', {})
        self.triangle_height = int(triangle_config.get('height', 30))
        self.triangle_width = int(triangle_config.get('width', 20))
        self.triangle_x_offset = int(triangle_config.get('x_offset', 1))
        self.triangle_y_offset = int(triangle_config.get('y_offset', 15))
        self.triangle_color = self.parse_color(triangle_config.get('color', '(0,0,255)'))
        
        self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)
        self.current_smooth_position = None
        self.recent_positions = deque(maxlen=RECENT_POSITIONS_LENGTH)
        
        self.model = YOLO(model_path)
        self.video_info = sv.VideoInfo.from_video_path(source_video)
        self.frame_generator = sv.get_video_frames_generator(source_video)
        self.output_path = output_path
        self.frames_without_detection = 0

        self.kalman = None
        self.kalman_initialized = False
        self.video_writer = None
        self.last_valid_position = None

        # Trail parameters
        self.trail_length = trail_length
        self.trail_width = trail_width
        self.trail_speed = trail_speed
        self.trail_color = self.parse_color(trail_color)
        self.trail_positions = deque(maxlen=trail_length)
        self.trail_alpha = np.linspace(0.2, 1.0, trail_length)

        print(f"Trail parameters: {trail_length}, {trail_width}, {trail_speed}, {trail_color}")
        
        # EMA parameters
        self.ema_position = None
        self.ema_positions = deque(maxlen=trail_length)
        self.ema_alpha = ema_alpha
        
        # Progress callback
        self.progress_callback = None

    @staticmethod
    def parse_color(color_str: str) -> tuple:
        """Parse color string from JSON into BGR tuple"""
        try:
            # Remove parentheses and split by comma
            color_values = color_str.strip('()').split(',')
            # Convert to integers and return as BGR tuple
            return tuple(map(int, color_values))
        except:
            return (0, 0, 255)  # Default to red if parsing fails
        
    @staticmethod
    def load_config(config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file: {str(e)}")
            return {}


    def update_trail_parameters(self, trail_length: int = None, 
                              trail_width: int = None, 
                              trail_speed: float = None):
        """Update trail parameters dynamically"""
        if trail_length is not None:
            self.trail_length = trail_length
            self.trail_positions = deque(maxlen=trail_length)
            self.trail_alpha = np.linspace(0.2, 1.0, trail_length)
            self.ema_positions = deque(maxlen=trail_length)
        if trail_width is not None:
            self.trail_width = trail_width
        if trail_speed is not None:
            self.trail_speed = trail_speed

    def calculate_ema(self, position: np.ndarray) -> np.ndarray:
        """Calculate Exponential Moving Average for the given position"""
        if self.ema_position is None:
            self.ema_position = position
        else:
            self.ema_position = self.ema_alpha * position + (1 - self.ema_alpha) * self.ema_position
        return self.ema_position

    def draw_trail(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Draw the ball trail with EMA smoothing, reversed fade effect, and enhanced dynamics"""
        if len(position) > 0:
            # Shift the position slightly upward and apply EMA
            shifted_position = position.copy()
            shifted_position[1] -= 6  # Shift y-coordinate up
            smoothed_position = self.calculate_ema(shifted_position)
            self.ema_positions.append(smoothed_position)

        # Draw the smoothed trail with enhanced effects
        if len(self.ema_positions) >= 2:
            positions = np.array(self.ema_positions)
            trail_length = len(positions)
            
            # Draw from oldest to newest for proper layering
            for i in range(trail_length - 1, 0, -1):
                # Calculate relative position in trail (0 = oldest, 1 = newest)
                rel_pos = i / (trail_length - 1)
                
                # Enhanced dynamic thickness
                base_thickness = self.trail_width + 2
                thickness = int(base_thickness * (0.2 + 0.8 * rel_pos))
                
                # Enhanced opacity calculation with smoother falloff
                alpha = 0.1 + 0.9 * (rel_pos ** 1.5)  # Exponential falloff
                
                # Get points for current segment
                pt1 = positions[i-1].astype(np.int32)
                pt2 = positions[i].astype(np.int32)
                
                # Calculate segment velocity for color variation
                velocity = np.linalg.norm(pt2 - pt1)
                max_velocity = 30  # Adjust based on your needs
                velocity_factor = min(velocity / max_velocity, 1.0)
                
                # Dynamic color based on velocity
                r, g, b = self.trail_color
                velocity_color = (
                    int(r * (1 - 0.5 * velocity_factor)),  # Reduce red with velocity
                    int(g + (255 - g) * velocity_factor),  # Increase green with velocity
                    int(b * (1 - 0.3 * velocity_factor))   # Slightly reduce blue
                )
                
                # Draw main trail
                overlay = frame.copy()
                cv2.line(overlay, tuple(pt1), tuple(pt2), 
                        velocity_color, thickness, 
                        cv2.LINE_AA)
                
                # Add glow effect for high velocity segments
                if velocity_factor > 0.5:
                    glow_alpha = alpha * velocity_factor * 0.5
                    glow_thickness = thickness + 4
                    cv2.line(overlay, tuple(pt1), tuple(pt2),
                            (255, 255, 255), glow_thickness,
                            cv2.LINE_AA)
                    cv2.addWeighted(frame, 1 - glow_alpha, overlay, glow_alpha, 0, frame)
                
                # Add main trail segment
                cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
                
                # Optional: Add motion blur effect for fast segments
                if velocity_factor > 0.7:
                    blur_region = cv2.GaussianBlur(
                        frame[
                            max(0, min(pt1[1], pt2[1])-10):min(frame.shape[0], max(pt1[1], pt2[1])+10),
                            max(0, min(pt1[0], pt2[0])-10):min(frame.shape[1], max(pt1[0], pt2[0])+10)
                        ],
                        (5, 5), 2
                    )
                    frame[
                        max(0, min(pt1[1], pt2[1])-10):min(frame.shape[0], max(pt1[1], pt2[1])+10),
                        max(0, min(pt1[0], pt2[0])-10):min(frame.shape[1], max(pt1[0], pt2[0])+10)
                    ] = blur_region

        return frame


    def draw_triangle_indicator(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Draw a small filled inverted triangle above the ball with configurable parameters"""
        if len(position) == 0:
            return frame

        try:
            # Calculate triangle vertices using config parameters
            ball_x, ball_y = int(position[0]), int(position[1])

            horizonal_offset = self.triangle_x_offset + 1
            vertical_offset = self.triangle_y_offset
            triangle_width = self.triangle_width
            triangle_height = self.triangle_height // 2
            
            # Apply offsets from config
            top_point = (
                ball_x + horizonal_offset,
                ball_y - vertical_offset
            )
            left_point = (
                ball_x - triangle_width // 2 + horizonal_offset,
                ball_y - triangle_height - vertical_offset
            )
            right_point = (
                ball_x + triangle_width // 2 + horizonal_offset,
                ball_y - triangle_height - vertical_offset
            )

            print(f"Triangle points: {top_point}, {left_point}, {right_point}")
            
            # Draw filled inverted triangle with configured color
            pts = np.array([top_point, left_point, right_point], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], self.triangle_color, cv2.LINE_AA)
            
            return frame

        except Exception as e:
            print(f"Warning: Failed to draw filled inverted triangle: {str(e)}")
            return frame


    def setup_kalman_filter(self):
        """Initialize a new Kalman filter"""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.01
        
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                   [0, 1]], np.float32) * 0.1
        
    def initialize_kalman_state(self, measurement: np.ndarray):
        """Initialize Kalman filter state with first detection"""
        try:
            self.kalman.statePre = np.array([[measurement[0]],
                                           [measurement[1]],
                                           [0],
                                           [0]], np.float32)
            self.kalman.statePost = self.kalman.statePre.copy()
            self.kalman_initialized = True
            self.frames_without_detection = 0
        except Exception as e:
            print(f"Warning: Failed to initialize Kalman filter: {str(e)}")
            self.kalman_initialized = False

    def smooth_position(self, position: np.ndarray) -> np.ndarray:
        """Apply position smoothing using exponential moving average"""
        if len(position) == 0:
            return position

        # Initialize smooth position if needed
        if self.current_smooth_position is None:
            self.current_smooth_position = position
            return position

        # Apply exponential smoothing
        self.current_smooth_position = (
            SMOOTHING_FACTOR * position +
            (1 - SMOOTHING_FACTOR) * self.current_smooth_position
        )

        # Add to position history
        self.position_history.append(self.current_smooth_position)

        # Apply additional smoothing if we have enough history
        if len(self.position_history) >= 3:
            # Use Savitzky-Golay filter for additional smoothing
            positions_array = np.array(self.position_history)
            window_length = min(len(self.position_history), 5)
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                try:
                    smoothed = savgol_filter(positions_array, window_length, 2, axis=0)
                    self.current_smooth_position = smoothed[-1]
                except ValueError:
                    pass  # Fall back to exponential smoothing if Savitzky-Golay fails

        return self.current_smooth_position



    def predict_ball_position(self, measurement: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        """Predict ball position with Kalman filtering"""
        try:
            if measurement.size > 0:
                self.frames_without_detection = 0
                
                consistent_detections = sum(1 for pos in self.recent_positions if len(pos) > 0)
                if consistent_detections >= 2:
                    if not self.kalman_initialized:
                        self.setup_kalman_filter()
                        self.initialize_kalman_state(measurement)
                        raw_prediction = measurement
                    else:
                        prediction = self.kalman.predict()
                        measurement_noise = 0.05 / max(confidence + 0.1, 0.01)
                        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
                        self.kalman.correct(measurement.reshape(-1, 1))
                        self.last_valid_position = measurement
                        raw_prediction = prediction[:2].reshape(-1)
                else:
                    return np.array([])
            else:
                self.frames_without_detection += 1
                if self.frames_without_detection >= MAX_FRAMES_WITHOUT_DETECTION:
                    self.kalman_initialized = False
                    self.current_smooth_position = None
                    return np.array([])
                
                if self.kalman_initialized:
                    prediction = self.kalman.predict()
                    raw_prediction = prediction[:2].reshape(-1)
                else:
                    return np.array([])

            # Apply smoothing to the raw prediction
            smoothed_position = self.smooth_position(raw_prediction)
            return smoothed_position

        except Exception as e:
            print(f"Warning: Position prediction failed: {str(e)}")
            return np.array([])

    
    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if position is within frame bounds"""
        try:
            x, y = position
            return (0 <= x < self.video_info.width and 
                   0 <= y < self.video_info.height)
        except:
            return False


    def apply_motion_blur_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply motion blur detection to help identify fast-moving balls"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.addWeighted(gray, 1.5, motion_blur, -0.5, 0)

    def enhance_ball_visibility(self, frame: np.ndarray) -> np.ndarray:
        """Enhance ball visibility using image processing techniques"""
        # Convert to HSV and enhance brightness/contrast
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        return enhanced

    def detect_ball(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect ball using both regular and enhanced frames with outlier filtering"""
        # Try detection on original frame
        result = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(result)

        # If no ball found, try on enhanced frame
        if len(detections[detections.class_id == BALL_ID]) == 0:
            enhanced_frame = self.enhance_ball_visibility(frame)
            result = self.model.predict(enhanced_frame, conf=CONFIDENCE_THRESHOLD)[0]
            detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]

        if len(ball_detections) == 0:
            self.recent_positions.append(np.array([]))
            return np.array([]), 0.0

        # Get coordinates and confidence of the highest confidence detection
        coords = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        confidence = ball_detections.confidence[0]
        
        # Add the new position to recent positions
        self.recent_positions.append(coords[0])
        
        # Apply outlier detection
        cleaned_positions = replace_outliers_based_on_distance(
            list(self.recent_positions),
            DISTANCE_THRESHOLD
        )
        
        # Get the latest cleaned position
        latest_cleaned_position = cleaned_positions[-1]
        
        if len(latest_cleaned_position) == 0:
            return np.array([]), 0.0
            
        return latest_cleaned_position, confidence

    def smooth_trajectory(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """Apply Savitzky-Golay filter to smooth the trajectory"""
        if len(positions) < MAXLEN:
            return positions

        valid_positions = [pos for pos in positions if len(pos) > 0]
        if len(valid_positions) < MAXLEN:
            return positions

        positions_array = np.array(valid_positions)
        smoothed_x = savgol_filter(positions_array[:, 0], MAXLEN, 3)
        smoothed_y = savgol_filter(positions_array[:, 1], MAXLEN, 3)

        smoothed_positions = []
        valid_idx = 0
        for pos in positions:
            if len(pos) > 0:
                smoothed_positions.append(np.array([smoothed_x[valid_idx],
                                                  smoothed_y[valid_idx]]))
                valid_idx += 1
            else:
                smoothed_positions.append(np.array([]))

        return smoothed_positions

    def process_video(self):
        """Process video with error handling and resource cleanup"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc,
                self.video_info.fps,
                (self.video_info.width, self.video_info.height)
            )

            if not self.video_writer.isOpened():
                raise RuntimeError("Failed to create output video file")

            total_frames = self.video_info.total_frames
            start_time = time.time()  # Track start time

            for i, frame in enumerate(tqdm(self.frame_generator, total=total_frames, desc="Processing", unit="frame")):
                self._process_frame(frame)
                if self.progress_callback:
                    elapsed_time = time.time() - start_time
                    remaining_time = (total_frames - (i + 1)) * (elapsed_time / (i + 1)) if i > 0 else None
                    self.progress_callback(i + 1, total_frames, remaining_time)

        except Exception as e:
            raise RuntimeError(f"Video processing failed: {str(e)}")
        finally:
            if self.video_writer is not None:
                self.video_writer.release()

    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
        # Initialize tqdm instance for later use in time estimation
        self.tqdm_instance = tqdm(total=self.video_info.total_frames, desc="Processing", unit="frame", leave=False, position=0)


    def _process_frame(self, frame: np.ndarray):
        """Process single frame with trail and triangle indicator"""
        try:
            ball_coords, confidence = self.detect_ball(frame)
            predicted_position = self.predict_ball_position(ball_coords, confidence)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw trail first (so it appears behind the ball and triangle)
            if len(predicted_position) > 0:
                annotated_frame = self.draw_trail(annotated_frame, predicted_position)
                
                # Draw the triangle
                annotated_frame = self.draw_triangle_indicator(annotated_frame, predicted_position)
                
            self.video_writer.write(annotated_frame)

        except Exception as e:
            print(f"Warning: Frame processing failed: {str(e)}")    

# Modify your main() function:
def main():
    app = VideoProcessorGUI()
    app.run()

if __name__ == "__main__":
    main()