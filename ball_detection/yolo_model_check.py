from ultralytics import YOLO
import torch

# Load the YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("/home/sahil/Downloads/ball-tracker-cv/football-ball-detection-v2.pt")
model.to(device)


# Perform prediction with confidence threshold and save results
results = model.predict("/home/sahil/Downloads/goal_0002.avi", conf=0.1, save=True)

# Loop through the results (since model.predict returns a list)
for result in results:
    # Access bounding box coordinates (xyxy format) for the first detection
    if len(result.boxes) > 0:  # Check if there are any detections
        box = result.boxes[0].xyxy[0].cpu().numpy()
        print("Bounding Box Coordinates:", box)
    else:
        print("No detections found in the frame.")
