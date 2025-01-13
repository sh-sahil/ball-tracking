import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import deque
from typing import List, Union, Tuple, Optional
from scipy.signal import savgol_filter
from tqdm import tqdm
import torch
import os
from pathlib import Path
import imageio
from PIL import Image

BALL_ID = 0  # Class ID for the ball
MAXLEN = 15  # Increased window size for better smoothing
DISTANCE_THRESHOLD = 80.0  # Distance threshold for outlier detection
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for ball detection 
MAX_FRAMES_WITHOUT_DETECTION = 8  # Maximum frames to show prediction without detection
POSITION_HISTORY_SIZE = 6  # Number of positions to keep for smoothing
SMOOTHING_FACTOR = 0.6    # Weight for current position (0 to 1)

TRAIL_LENGTH = 30  # Number of previous positions to keep for trail
TRAIL_FADE_FACTOR = 0.95  # How quickly the trail fades (0-1)
MIN_TRAIL_POINTS = 5  # Minimum points needed to draw a trail
TRAIL_SPEED_FACTOR = 1.0 # Control trail speed (adjust between 0.1 and 2.0)
RECENT_POSITIONS_LENGTH = 8  # Number of recent positions to keep for Kalman filter

SOURCE_VIDEO_PATH = "/home/sahil/Desktop/Sportvot/Goal - ARIS KHAN.mp4"
OUTPUT_VIDEO_PATH = "/home/sahil/Desktop/Sportvot/Ouput-Goal - ARIS KHAN.mp4"
PLAYER_MODEL_PATH = "/media/sahil/UBUNTU/best-100.pt"  # Add your model path here
GIF_ASSET_PATH = "/home/sahil/Downloads/explosion-12389_256.gif"  # Add your asset path here


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
    def __init__(self, model_path: str, source_video: str, output_path: str, gif_path: str, trail_speed: float = 1.0):
        # Previous initialization remains the same
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Video file not found: {source_video}")
        
        self.position_history = deque(maxlen=POSITION_HISTORY_SIZE)
        self.trail_positions = deque(maxlen=TRAIL_LENGTH)  # Add trail position history
        self.current_smooth_position = None
        self.recent_positions = deque(maxlen=RECENT_POSITIONS_LENGTH)
        self.trail_speed = trail_speed # Add trail speed factor
        
        self.model = YOLO(model_path)
        self.video_info = sv.VideoInfo.from_video_path(source_video)
        self.frame_generator = sv.get_video_frames_generator(source_video)
        self.output_path = output_path
        self.frames_without_detection = 0
        self.kalman = None
        self.kalman_initialized = False
        self.video_writer = None
        self.last_valid_position = None
        self.gif_frames = []
        self.current_frame_idx = 0
        
        # Load GIF frames as before
        try:
            gif = imageio.get_reader(gif_path)
            for frame in gif:
                pil_frame = Image.fromarray(frame).convert('RGBA')
                pil_frame = pil_frame.resize((40, 40), Image.Resampling.LANCZOS)
                np_frame = np.array(pil_frame)
                self.gif_frames.append(np_frame)
        except Exception as e:
            print(f"Failed to load GIF: {str(e)}")
            self.gif_frames = None

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
    


    def draw_trail(self, frame: np.ndarray) -> np.ndarray:
        """Draw a smooth trailing effect behind the ball with speed control"""
        if len(self.trail_positions) < MIN_TRAIL_POINTS:
            return frame
            
        try:
            # Apply speed factor to control trail length
            effective_trail_length = max(MIN_TRAIL_POINTS, 
                                      int(len(self.trail_positions) * (1.0 / self.trail_speed)))
            
            # Convert positions to numpy array for processing
            all_points = np.array([[int(pos[0]), int(pos[1])] 
                                 for pos in self.trail_positions if len(pos) > 0])
            
            if len(all_points) < MIN_TRAIL_POINTS:
                return frame
            
            # Sample points based on trail speed
            points = all_points[::max(1, int(1 * self.trail_speed))]
            
            # Create a copy of the frame for blending
            overlay = frame.copy()
            
            # Draw the trail with varying thickness and alpha
            for i in range(1, len(points)):
                # Calculate alpha value that decreases for older positions
                alpha = (i / len(points)) * TRAIL_FADE_FACTOR
                
                # Calculate thickness that decreases for older positions
                thickness = max(1, int(3 * (i / len(points))))
                
                # Draw line segment with anti-aliasing
                cv2.line(
                    overlay,
                    tuple(points[i-1]),
                    tuple(points[i]),
                    (255, 255, 255),  # White color
                    thickness,
                    cv2.LINE_AA
                )
            
            # Blend the trail with the original frame
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            return frame
            
        except Exception as e:
            print(f"Warning: Failed to draw trail: {str(e)}")
            return frame


    def predict_ball_position(self, measurement: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        """Predict ball position with improved responsiveness"""
        try:
            if measurement.size > 0:
                self.frames_without_detection = 0
                
                # Reduced threshold for consistent detections
                consistent_detections = sum(1 for pos in self.recent_positions if len(pos) > 0)
                if consistent_detections >= 2:  # Reduced from 3
                    if not self.kalman_initialized:
                        self.setup_kalman_filter()
                        self.initialize_kalman_state(measurement)
                        raw_prediction = measurement
                    else:
                        prediction = self.kalman.predict()
                        # Increased measurement weight for better responsiveness
                        measurement_noise = 0.05 / max(confidence + 0.1, 0.01)  # Reduced from 0.1
                        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise
                        self.kalman.correct(measurement.reshape(-1, 1))
                        self.last_valid_position = measurement
                        raw_prediction = prediction[:2].reshape(-1)
                        
                    # Add position to trail considering speed factor
                    if len(self.trail_positions) > 0:
                        last_pos = self.trail_positions[-1]
                        if len(last_pos) > 0:
                            dist = np.linalg.norm(raw_prediction - last_pos)
                            if dist > (2.0 * self.trail_speed):  # Adjust spacing based on speed
                                self.trail_positions.append(raw_prediction)
                    else:
                        self.trail_positions.append(raw_prediction)
                else:
                    return np.array([])
            else:
                self.frames_without_detection += 1
                if self.frames_without_detection >= MAX_FRAMES_WITHOUT_DETECTION:
                    self.kalman_initialized = False
                    self.current_smooth_position = None
                    self.trail_positions.clear()
                    return np.array([])
                
                if self.kalman_initialized:
                    prediction = self.kalman.predict()
                    raw_prediction = prediction[:2].reshape(-1)
                    self.trail_positions.append(raw_prediction)
                else:
                    return np.array([])

            # Apply smoothing to the raw prediction
            smoothed_position = self.smooth_position(raw_prediction)
            return smoothed_position

        except Exception as e:
            print(f"Warning: Position prediction failed: {str(e)}")
            return np.array([])

    
    def overlay_tracker_asset(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Overlay the fire GIF animation with motion blur effect"""
        if len(position) == 0 or self.gif_frames is None or len(self.gif_frames) == 0:
            return frame
            
        try:
            gif_frame = self.gif_frames[self.current_frame_idx]
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.gif_frames)
            
            # Adjust the vertical offset here by changing the division factor
            # Increase the number to move the ball up, decrease to move it down
            vertical_offset = gif_frame.shape[0] * 0.75  # Changed from /2 to *0.75 to move it up
            
            # Calculate position with adjusted vertical offset
            x = int(position[0] - gif_frame.shape[1]/2)
            y = int(position[1] - vertical_offset)  # Using the new vertical offset
            center_x = int(position[0])
            center_y = int(position[1])
            
            if self.last_valid_position is not None:
                last_x, last_y = self.last_valid_position
                if abs(center_x - last_x) > 5 or abs(center_y - last_y) > 5:
                    pts = np.array([[center_x, center_y], [last_x, last_y]], np.int32)
                    cv2.polylines(frame, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.GaussianBlur(frame, (3, 3), 0, frame)
            
            self.last_valid_position = np.array([center_x, center_y])
            
            if x < 0 or y < 0 or x + gif_frame.shape[1] > frame.shape[1] or y + gif_frame.shape[0] > frame.shape[0]:
                return frame
            
            alpha = gif_frame[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(gif_frame[:, :, :3], cv2.COLOR_RGB2BGR)
            
            roi = frame[y:y+gif_frame.shape[0], x:x+gif_frame.shape[1]]
            blended = roi * (1 - alpha) + bgr * alpha
            frame[y:y+gif_frame.shape[0], x:x+gif_frame.shape[1]] = blended.astype(np.uint8)
                    
        except Exception as e:
            print(f"Warning: Failed to overlay GIF frame: {str(e)}")
                
        return frame
    
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

            for frame in tqdm(self.frame_generator,
                            total=self.video_info.total_frames):
                self._process_frame(frame)

        except Exception as e:
            raise RuntimeError(f"Video processing failed: {str(e)}")
        finally:
            if self.video_writer is not None:
                self.video_writer.release()


    def _process_frame(self, frame: np.ndarray):
        """Process single frame with improved tracking and trail effect"""
        try:
            ball_coords, confidence = self.detect_ball(frame)
            predicted_position = self.predict_ball_position(ball_coords, confidence)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw trail effect first (so it appears behind the ball)
            annotated_frame = self.draw_trail(annotated_frame)
            
            # Only draw if we have a valid position
            if len(predicted_position) > 0:
                # Overlay tracker asset instead of drawing circles
                annotated_frame = self.overlay_tracker_asset(annotated_frame, predicted_position)
                
                # Add confidence score if we have a detection
                if len(ball_coords) > 0:
                    cv2.putText(
                        annotated_frame,
                        f"Conf: {confidence:.2f}",
                        (int(predicted_position[0]) + 20, int(predicted_position[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
                
                # Add prediction indicator when using predicted position
                if len(ball_coords) == 0 and self.frames_without_detection < MAX_FRAMES_WITHOUT_DETECTION:
                    cv2.putText(
                        annotated_frame,
                        f"Predicted ({self.frames_without_detection} frames)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            
            self.video_writer.write(annotated_frame)

        except Exception as e:
            print(f"Warning: Frame processing failed: {str(e)}")

    def _add_prediction_indicator(self, frame: np.ndarray):
        """Add prediction indicator with error handling"""
        try:
            cv2.putText(
                frame,
                f"Predicted ({self.frames_since_detection} frames)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        except Exception as e:
            print(f"Warning: Failed to add prediction indicator: {str(e)}")

    def annotate_frame(self, frame: np.ndarray, ball_coords: np.ndarray,
                      predicted_position: np.ndarray,
                      smoothed_path: List[np.ndarray],
                      confidence: float) -> np.ndarray:
        """Annotate frame with ball position and trajectory"""
        annotated_frame = frame.copy()

        # Draw predicted position
        cv2.circle(
            annotated_frame,
            center=(int(predicted_position[0]), int(predicted_position[1])),
            radius=5,
            color=(0, 255, 0),  # Green for prediction
            thickness=1
        )

        # Draw actual ball position if detected
        if len(ball_coords) > 0:
            cv2.circle(
                annotated_frame,
                center=(int(ball_coords[0]), int(ball_coords[1])),
                radius=5,
                color=(0, 0, 255),  # Red for actual detection
                thickness=-1
            )

            # Add confidence score
            cv2.putText(
                annotated_frame,
                f"Conf: {confidence:.2f}",
                (int(ball_coords[0]) + 10, int(ball_coords[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        # Draw smoothed trajectory
        for i in range(1, len(smoothed_path)):
            if len(smoothed_path[i]) > 0 and len(smoothed_path[i - 1]) > 0:
                cv2.line(
                    annotated_frame,
                    pt1=(int(smoothed_path[i - 1][0]),
                         int(smoothed_path[i - 1][1])),
                    pt2=(int(smoothed_path[i][0]),
                         int(smoothed_path[i][1])),
                    color=(255, 255, 0),  # Yellow for trajectory
                    thickness=2
                )

        return annotated_frame

def main():
    tracker = BallTracker(
        model_path=PLAYER_MODEL_PATH,
        source_video=SOURCE_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        gif_path=GIF_ASSET_PATH,
        trail_speed=1.0
    )
    tracker.process_video()

if __name__ == "__main__":
    main()