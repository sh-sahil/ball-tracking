import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import deque
from typing import List, Union, Tuple
from scipy.signal import savgol_filter
from tqdm import tqdm
import os


# Basic tracking parameters
BALL_ID = 0
MAXLEN = 15
DISTANCE_THRESHOLD = 80.0
CONFIDENCE_THRESHOLD = 0.3
MAX_FRAMES_WITHOUT_DETECTION = 8
POSITION_HISTORY_SIZE = 6
SMOOTHING_FACTOR = 0.6
RECENT_POSITIONS_LENGTH = 8

# Triangle parameters (new)
TRIANGLE_HEIGHT = 30  # Height above the ball
TRIANGLE_WIDTH = 20   # Width of the triangle base
TRIANGLE_COLOR = (0, 0, 255)  # Green color (BGR)
TRIANGLE_THICKNESS = 2  # Line thickness

# Trail parameters (new)
TRAIL_LENGTH = 20  # Number of previous positions to use
TRAIL_WIDTH = 3    # Width of the trail in pixels
TRAIL_SPEED = 1    # Speed factor for trail fade (1 = normal, 2 = faster, 0.5 = slower)
TRAIL_COLOR = (255, 255, 0)  # Yellow color in BGR
EMA_ALPHA = 0.3    # Exponential moving average smoothing factor (0-1)


SOURCE_VIDEO_PATH = "/home/sahil/Desktop/Sportvot/Videos/Goal - ARIS KHAN.mp4"
OUTPUT_VIDEO_PATH = "/home/sahil/Desktop/Sportvot/Videos/EMA.mp4"
PLAYER_MODEL_PATH = "/home/sahil/Desktop/Sportvot/best-100.pt"  # Add your model path here


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
                 triangle_height: int = 30, triangle_width: int = 20,
                 trail_length: int = TRAIL_LENGTH,
                 trail_width: int = TRAIL_WIDTH,
                 trail_speed: float = TRAIL_SPEED):
        # Initialization code remains the same...
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Video file not found: {source_video}")
        
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
        
        # Triangle parameters
        self.triangle_height = triangle_height
        self.triangle_width = triangle_width

        # Trail parameters
        self.trail_length = trail_length
        self.trail_width = trail_width
        self.trail_speed = trail_speed
        self.trail_positions = deque(maxlen=trail_length)
        self.trail_alpha = np.linspace(0.2, 1.0, trail_length)
        
        # EMA parameters
        self.ema_position = None
        self.ema_positions = deque(maxlen=trail_length)


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
            self.ema_position = EMA_ALPHA * position + (1 - EMA_ALPHA) * self.ema_position
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
                r, g, b = TRAIL_COLOR
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
        """Draw a small filled inverted triangle above the ball, shifted upward"""
        if len(position) == 0:
            return frame

        try:
            # Calculate triangle vertices
            ball_x, ball_y = int(position[0]), int(position[1])
            
            # Parameters for triangle size and offset
            horizontal_offset = 1  # Shift triangle 3 pixels to the right
            offset = 15  # Pixels to push the triangle further up
            triangle_width = self.triangle_width  # Increase base width
            triangle_height = self.triangle_height // 2  # Maintain height size
            
            # Adjusted triangle points for inverted triangle above the ball
            top_point = (ball_x + horizontal_offset, ball_y - offset)  # Point closer to the ball
            left_point = (ball_x - triangle_width // 2 + horizontal_offset, ball_y - triangle_height - offset)
            right_point = (ball_x + triangle_width // 2 + horizontal_offset, ball_y - triangle_height - offset)
            
            # Draw filled inverted triangle
            pts = np.array([top_point, left_point, right_point], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], TRIANGLE_COLOR, cv2.LINE_AA)  # Green color
            
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

            for frame in tqdm(self.frame_generator,
                            total=self.video_info.total_frames):
                self._process_frame(frame)

        except Exception as e:
            raise RuntimeError(f"Video processing failed: {str(e)}")
        finally:
            if self.video_writer is not None:
                self.video_writer.release()


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

def main():
    # Create tracker with customizable parameters
    tracker = BallTracker(
        model_path=PLAYER_MODEL_PATH,
        source_video=SOURCE_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
    )
    
    tracker.process_video()

if __name__ == "__main__":
    main()