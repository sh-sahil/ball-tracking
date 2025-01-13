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

BALL_ID = 0  # Class ID for the ball
MAXLEN = 15  # Increased window size for better smoothing
DISTANCE_THRESHOLD = 50.0  # Distance threshold for outlier detection
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for ball detection
TRACKING_HISTORY = 30  # Number of frames to keep in tracking history
MAX_PREDICTION_FRAMES = 10  # Maximum number of frames to predict without detection

SOURCE_VIDEO_PATH = "/home/sahil/Downloads/Goal - ARIS KHAN.mp4"
OUTPUT_VIDEO_PATH = "/home/sahil/Desktop/Sportvot/Output60.mp4"
PLAYER_MODEL_PATH = "/home/sahil/Downloads/ball-tracker-cv/football-ball-detection-v2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def replace_outliers_based_on_distance(
    positions: List[np.ndarray],
    distance_threshold: float
) -> List[np.ndarray]:
    last_valid_position: Union[np.ndarray, None] = None
    cleaned_positions: List[np.ndarray] = []

    for position in positions:
        if len(position) == 0:
            # If the current position is already empty, just add it to the cleaned positions
            cleaned_positions.append(position)
        else:
            if last_valid_position is None:
                # If there's no valid last position, accept the first valid one
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                # Calculate the distance from the last valid position
                distance = np.linalg.norm(position - last_valid_position)
                if distance > distance_threshold:
                    # Replace with empty array if the distance exceeds the threshold
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    cleaned_positions.append(position)
                    last_valid_position = position

    return cleaned_positions

class BallTracker:
    def __init__(self, model_path: str, source_video: str, output_path: str):
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"Video file not found: {source_video}")
        
        # Validate output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
            
        self.model = YOLO(model_path)
        self.model.to(device)
        try:
            self.video_info = sv.VideoInfo.from_video_path(source_video)
            self.frame_generator = sv.get_video_frames_generator(source_video)
        except Exception as e:
            raise RuntimeError(f"Failed to open video file: {str(e)}")
            
        self.output_path = output_path
        self.path_raw = deque(maxlen=1000)  # Limit stored positions
        self.tracking_history = deque(maxlen=TRACKING_HISTORY)
        self.frames_since_detection = 0
        self.kalman = cv2.KalmanFilter(4, 2)
        self.setup_kalman_filter()
        self.kalman_initialized = False
        self.video_writer = None

    def setup_kalman_filter(self):
        """Initialize Kalman filter with improved parameters for ball tracking"""
        # State: [x, y, dx, dy]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        
        # Include velocity in state transition
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        
        # Tune process noise for smoother predictions
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.01
        
        # Measurement noise - adjust based on detection confidence
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                   [0, 1]], np.float32) * 0.1
        
    def initialize_kalman_state(self, measurement: np.ndarray):
        """Initialize Kalman filter state with first detection and estimated velocity"""
        try:
            if len(self.tracking_history) >= 2:
                # Estimate initial velocity from last two positions
                prev_pos = self.tracking_history[-2]
                curr_pos = self.tracking_history[-1]
                initial_velocity = curr_pos - prev_pos
            else:
                initial_velocity = np.array([0, 0])

            self.kalman.statePre = np.array([[measurement[0]],
                                           [measurement[1]],
                                           [initial_velocity[0]],
                                           [initial_velocity[1]]], np.float32)
            self.kalman.statePost = self.kalman.statePre.copy()
            self.kalman_initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize Kalman filter: {str(e)}")
            self.kalman_initialized = False

    def predict_ball_position(self, measurement: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        """Predict ball position with continuous tracking even without detection"""
        try:
            # Initialize Kalman filter with first valid measurement
            if not self.kalman_initialized and measurement.size > 0:
                self.initialize_kalman_state(measurement)
                return measurement

            # Always predict next state
            prediction = self.kalman.predict()
            pred_pos = prediction[:2].reshape(-1)
            
            # If no measurement, increment counter but keep predicting
            if measurement.size == 0:
                self.frames_since_detection += 1
                
                # Validate prediction is within bounds
                if not self._is_position_valid(pred_pos):
                    # If prediction is out of bounds, use last valid position
                    return self.path_raw[-1] if len(self.path_raw) > 0 else np.array([])
                    
                return pred_pos

            # When we have a measurement, update the filter
            measurement_noise = 0.1 / max(confidence + 0.1, 0.01)
            self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * measurement_noise

            measurement = measurement.reshape(-1, 1)
            self.kalman.correct(measurement)
            self.frames_since_detection = 0
            
            return pred_pos
            
        except Exception as e:
            print(f"Warning: Kalman prediction failed: {str(e)}")
        return measurement if measurement.size > 0 else np.array([])
        
    
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
        """Detect ball using both regular and enhanced frames"""
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
            return np.array([]), 0.0

        # Return coordinates and confidence of the highest confidence detection
        coords = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        return coords[0], ball_detections.confidence[0]

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
        """Process single frame maintaining continuous tracking"""
        try:
            ball_coords, confidence = self.detect_ball(frame)
            
            # Always get a prediction, whether we have detection or not
            predicted_position = self.predict_ball_position(ball_coords, confidence)
            
            # Update tracking history with either detection or prediction
            if len(ball_coords) > 0:
                self.tracking_history.append(ball_coords)
                self.path_raw.append(ball_coords)
            elif len(predicted_position) > 0:
                # Use prediction when no detection available
                self.tracking_history.append(predicted_position)
                self.path_raw.append(predicted_position)
            else:
                # If both detection and prediction fail, use last known position
                last_position = (self.path_raw[-1] if len(self.path_raw) > 0 
                            else np.array([]))
                self.tracking_history.append(last_position)
                self.path_raw.append(last_position)

            # Clean and smooth the trajectory
            cleaned_path = replace_outliers_based_on_distance(list(self.path_raw),
                                                            DISTANCE_THRESHOLD)
            smoothed_path = self.smooth_trajectory(cleaned_path)

            # Annotate frame with tracking information
            annotated_frame = self.annotate_frame(frame, ball_coords,
                                                predicted_position,
                                                smoothed_path, confidence)

            # Add prediction indicator when using predicted position
            if len(predicted_position) > 0 and len(ball_coords) == 0:
                self._add_prediction_indicator(annotated_frame)

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
        output_path=OUTPUT_VIDEO_PATH
    )
    tracker.process_video()

if __name__ == "__main__":
    main()