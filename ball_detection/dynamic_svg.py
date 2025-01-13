import cv2
import cairosvg
import svgwrite
import numpy as np

# Function to create dynamic SVG
def create_dynamic_svg(frame_num, width, height):
    dwg = svgwrite.Drawing(size=(width, height))
    # Example: Create a circle that changes size dynamically
    radius = 20 + (frame_num % 30)  # Radius oscillates between 20 and 50
    dwg.add(dwg.circle(center=(width // 2, height // 2), r=radius, fill="red"))
    return dwg.tostring()

# Function to render SVG to OpenCV image
def svg_to_image(svg_data, width, height):
    # Convert SVG data to PNG (byte array)
    png_data = cairosvg.svg2png(bytestring=svg_data, output_width=width, output_height=height)
    # Convert PNG byte data to OpenCV format
    png_array = np.frombuffer(png_data, dtype=np.uint8)
    image = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)
    return image

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with a video file path
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Example object detection (replace with real detection logic)
    height, width, _ = frame.shape
    detection_box = (width // 2 - 100, height // 2 - 100, width // 2 + 100, height // 2 + 100)
    cv2.rectangle(frame, detection_box[:2], detection_box[2:], (0, 255, 0), 2)

    # Simulate detection condition
    detected = True  # Replace this with real detection logic
    if detected:
        # Create a dynamic SVG for the current frame
        svg_data = create_dynamic_svg(frame_count, 200, 200)
        # Convert SVG to image
        svg_image = svg_to_image(svg_data, 200, 200)
        # Overlay the SVG image on the video frame
        x1, y1, x2, y2 = detection_box
        svg_resized = cv2.resize(svg_image, (x2 - x1, y2 - y1))
        # Extract alpha channel for blending
        alpha = svg_resized[:, :, 3] / 255.0  # Normalize alpha to 0-1
        for c in range(3):  # Blend RGB channels
            frame[y1:y2, x1:x2, c] = (alpha * svg_resized[:, :, c] +
                                      (1 - alpha) * frame[y1:y2, x1:x2, c])

    # Display the frame
    cv2.imshow("Dynamic SVG Animation on Detection", frame)
    frame_count += 1

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
