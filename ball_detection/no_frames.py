import cv2

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return -1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

video_path = "/home/sahil/Desktop/Sportvot/Goal.mp4"
frame_count = get_frame_count(video_path)
if frame_count != -1:
    print(f"The video has {frame_count} frames.")