import cv2

def convert_to_24fps(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    # Get the original video's properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps  # Calculate the original duration
    
    # Define the codec and create VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 files
    out = cv2.VideoWriter(output_video_path, codec, 24, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open output video writer.")
        cap.release()
        return
    
    # Calculate the frame selection ratio
    frame_interval = original_fps / 24
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write only the selected frames
        if current_frame % frame_interval < 1:
            out.write(frame)
        
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully converted to 24fps and saved to {output_video_path}")

# Example usage
input_video_path = '/home/sahil/Downloads/Screencast from 2025-01-06 19-01-45.mp4'
output_video_path = '/home/sahil/Desktop/Sportvot/segment_24.mp4'
convert_to_24fps(input_video_path, output_video_path)
