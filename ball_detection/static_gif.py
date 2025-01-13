import cv2
import imageio
import numpy as np
# Load the GIF using imageio
gif_path = "detection_animation.gif"
gif = imageio.mimread(gif_path)
# Convert GIF frames to OpenCV format
gif_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in gif]
# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with a video file path
frame_count = 0  # Frame counter for GIF animation
# Loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Example object detection (replace with your detection logic)
    # Simulate detection with a rectangle in the center
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    detection_box = (center_x - 50, center_y - 50, center_x + 50, center_y + 50)
    cv2.rectangle(frame, detection_box[:2], detection_box[2:], (0, 255, 0), 2)
    # Simulate detection condition (replace this with real detection logic)
    detected = True  # Replace with condition based on your detection logic
    if detected:
        # Get the current GIF frame
        gif_frame = gif_frames[frame_count % len(gif_frames)]
        frame_count += 1
        # Resize GIF frame to match the detection box size
        gif_frame_resized = cv2.resize(gif_frame, (100, 100))
        # Overlay GIF on the video frame
        x1, y1, x2, y2 = detection_box
        overlay = frame[y1:y2, x1:x2]
        combined = cv2.addWeighted(overlay, 0.5, gif_frame_resized, 0.5, 0)
        frame[y1:y2, x1:x2] = combined
    # Display the frame
    cv2.imshow("Object Detection with GIF", frame)
    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()