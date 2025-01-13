import cv2

# Load the video
video_path = "/home/sahil/Desktop/Sportvot/Goal.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

# Read the first frame to select ROI
ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read the first frame.")
    exit()

# Select ROI
roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Get ROI coordinates
x, y, w, h = map(int, roi)

# Define the codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
output_path = "cropped_video.mp4"  # Output file path
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width, frame_height = w, h
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop the ROI
    cropped_frame = frame[y:y+h, x:x+w]
    print("here")
    
    # Write the cropped frame to the output video
    out.write(cropped_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Cropped video saved as {output_path}")
