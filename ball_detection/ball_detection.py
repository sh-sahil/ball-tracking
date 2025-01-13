import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

class BallTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO("/media/sahil/UBUNTU/best-100.pt")  # Load YOLO model
        self.roi = None
        self.kalman = self._init_kalman()
        self.tracking_lost = False
        self.last_detection = None
        self.kalman_initialized = False
        self.consecutive_misses = 0
        self.max_misses = 10  # Maximum number of consecutive frames to predict without detection
        self.paused = False  # New flag for pause state

    def _init_kalman(self):
        """Initialize Kalman Filter for ball tracking"""
        kalman = KalmanFilter(dim_x=4, dim_z=2)
        
        kalman.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        kalman.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        kalman.R = np.eye(2) * 50  # Measurement noise
        kalman.Q = np.eye(4) * 0.1  # Process noise
        
        return kalman

    def _initialize_kalman_state(self, detection):
        """Initialize Kalman filter state with first detection"""
        self.kalman.x = np.array([detection[0], detection[1], 0., 0.])
        self.kalman.P = np.eye(4) * 100  # High initial uncertainty
        self.kalman_initialized = True
        self.consecutive_misses = 0

    def select_roi(self, frame):
        """Allow user to select initial ROI"""
        # Let user draw ROI
        roi = cv2.selectROI("Select Ball Region", frame, False)
        cv2.destroyWindow("Select Ball Region")
        
        self.roi = {
            'x': int(roi[0]),
            'y': int(roi[1]),
            'w': int(roi[2]),
            'h': int(roi[3])
        }
        
        # Reset tracking state
        self.kalman_initialized = False
        self.tracking_lost = False
        self.consecutive_misses = 0
        self.last_detection = None
        return True

    def update_roi(self, center):
        """Update ROI based on new ball position"""
        roi_size = 300  # Increased ROI size
        
        # Get frame dimensions
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Calculate new ROI coordinates with bounds checking
        x = max(0, min(int(center[0] - roi_size // 2), frame_width - roi_size))
        y = max(0, min(int(center[1] - roi_size // 2), frame_height - roi_size))
        
        self.roi = {
            'x': x,
            'y': y,
            'w': roi_size,
            'h': roi_size
        }

    def detect_ball(self, frame):
        """Detect ball using YOLO in current ROI"""
        if self.roi is None:
            return None

        try:
            # Ensure ROI bounds are within frame
            frame_height, frame_width = frame.shape[:2]
            self.roi['x'] = max(0, min(self.roi['x'], frame_width - self.roi['w']))
            self.roi['y'] = max(0, min(self.roi['y'], frame_height - self.roi['h']))
            
            # Extract ROI
            roi_frame = frame[
                self.roi['y']:self.roi['y'] + self.roi['h'],
                self.roi['x']:self.roi['x'] + self.roi['w']
            ]
            
            if roi_frame.size == 0:
                return None

            # Run YOLO detection on ROI
            results = self.model(roi_frame, classes=[32], conf=0.3)  # Lowered confidence threshold
            
            if len(results[0].boxes) > 0:
                # Get the detection with highest confidence
                box = results[0].boxes[0]
                confidence = box.conf.item()
                
                if confidence > 0.3:  # Confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert coordinates to full frame
                    center_x = self.roi['x'] + (x1 + x2) / 2
                    center_y = self.roi['y'] + (y1 + y2) / 2
                    
                    return np.array([center_x, center_y])
            
            return None
            
        except Exception as e:
            print(f"Error in ball detection: {e}")
            return None

    def track(self):
        """Main tracking loop"""
        ret, frame = self.cap.read()
        if not ret:
            return
            
        if not self.select_roi(frame):
            return

        frame_count = 0
        current_frame = frame

        while True:
            if not self.paused:
                ret, current_frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1

            ball_pos = self.detect_ball(current_frame)

            try:
                if ball_pos is not None:
                    if not self.kalman_initialized:
                        self._initialize_kalman_state(ball_pos)
                    else:
                        self.kalman.predict()
                        self.kalman.update(ball_pos)

                    self.last_detection = ball_pos
                    self.tracking_lost = False
                    self.consecutive_misses = 0
                    self.update_roi(ball_pos)
                else:
                    self.consecutive_misses += 1
                    
                    if self.kalman_initialized and self.consecutive_misses <= self.max_misses:
                        prediction = self.kalman.predict()
                        if prediction is not None:
                            predicted_pos = prediction[:2].flatten()
                            self.update_roi(predicted_pos)
                            self.tracking_lost = True
                    else:
                        # Reset tracking if too many consecutive misses
                        if self.consecutive_misses > self.max_misses:
                            self.kalman_initialized = False
                            self.tracking_lost = True

                self._visualize_frame(current_frame, frame_count)

            except Exception as e:
                print(f"Error in tracking loop: {e}")
                continue

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') and self.tracking_lost:
                self.paused = True
                self.select_roi(current_frame)
                self.paused = False

        self.cap.release()
        cv2.destroyAllWindows()

    def _visualize_frame(self, frame, frame_count):
        """Visualize tracking results"""
        # Draw ROI
        cv2.rectangle(frame, 
                     (self.roi['x'], self.roi['y']),
                     (self.roi['x'] + self.roi['w'], self.roi['y'] + self.roi['h']),
                     (0, 255, 0), 2)

        # Draw ball marker if detected
        if self.last_detection is not None and not self.tracking_lost:
            x, y = self.last_detection.astype(int)
            # Draw triangle marker above ball
            triangle_pts = np.array([
                [x, y - 20],
                [x - 10, y - 30],
                [x + 10, y - 30]
            ], np.int32)
            cv2.fillPoly(frame, [triangle_pts], (0, 0, 255))

        # Add status text
        status = f"Frame: {frame_count} | "
        status += "Tracking" if not self.tracking_lost else f"Lost (Misses: {self.consecutive_misses}) - Press 'p' to reselect ROI"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255) if self.tracking_lost else (0, 255, 0), 2)

        cv2.imshow("Ball Tracking", frame)

if __name__ == "__main__":
    tracker = BallTracker("/home/sahil/Desktop/Sportvot/ASHU.mp4")
    tracker.track()