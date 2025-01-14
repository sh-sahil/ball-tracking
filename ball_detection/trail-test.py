import cv2
import json
import numpy as np

# Constants from your code
DEFAULT_TRAIL_LENGTH = 20
DEFAULT_TRAIL_WIDTH = 3
DEFAULT_TRAIL_SPEED = 1
DEFAULT_TRAIL_COLOR = (255, 255, 0)  # BGR for yellow

# Functions from your main code (simplified for this example)
def draw_trail(frame, positions, trail_length, trail_width, trail_speed, trail_color):
    if len(positions) < 2:
        return frame

    overlay = frame.copy()
    alpha = trail_speed * np.linspace(0.1, 1.0, trail_length)
    
    for i in range(1, len(positions)):
        if i > len(positions) - trail_length:
            start = positions[i-1]
            end = positions[i]
            cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), trail_color, trail_width)
            cv2.addWeighted(frame, 1 - alpha[i - (len(positions) - trail_length)], overlay, alpha[i - (len(positions) - trail_length)], 0, frame)
    
    return frame

def main():
    # Define window size
    width, height = 640, 480
    cv2.namedWindow('Trail Test', cv2.WINDOW_NORMAL)

    # Initial fake points for simulation (generate a simple path)
    positions = []
    for i in range(100):
        x = 320 + int(100 * np.sin(i * 0.1))
        y = 240 + int(100 * np.cos(i * 0.1))
        positions.append(np.array([x, y]))

    # Parameters for the trail, can be adjusted via trackbars or direct input
    trail_length = DEFAULT_TRAIL_LENGTH
    trail_width = DEFAULT_TRAIL_WIDTH
    trail_speed = DEFAULT_TRAIL_SPEED
    trail_color = DEFAULT_TRAIL_COLOR

    # Create trackbars for parameter adjustment
    cv2.createTrackbar('Trail Length', 'Trail Test', trail_length, 50, lambda x: None)
    cv2.createTrackbar('Trail Width', 'Trail Test', trail_width, 10, lambda x: None)
    cv2.createTrackbar('Trail Speed', 'Trail Test', int(trail_speed * 10), 20, lambda x: None)

    while True:
        # Get updated values from trackbars
        trail_length = cv2.getTrackbarPos('Trail Length', 'Trail Test')
        trail_width = cv2.getTrackbarPos('Trail Width', 'Trail Test')
        trail_speed = cv2.getTrackbarPos('Trail Speed', 'Trail Test') / 10.0

        # Create a black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw the trail on the frame
        frame = draw_trail(frame, positions, trail_length, trail_width, trail_speed, trail_color)

        cv2.imshow('Trail Test', frame)

        # Save parameters to JSON
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
            params = {
                "TRAIL_LENGTH": trail_length,
                "TRAIL_WIDTH": trail_width,
                "TRAIL_SPEED": trail_speed,
                "TRAIL_COLOR": list(trail_color)
            }
            with open('trail_params.json', 'w') as json_file:
                json.dump(params, json_file)
            print("Parameters saved to trail_params.json")

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()