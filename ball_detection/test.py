import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import json

TRIANGLE_COLOR = (0, 255, 0)  # Green color for triangle


class BallTrackerApp:
    def __init__(self):
        # Initialize gif animation variables
        self.gif_frames = []
        self.current_frame_idx = 0
        self.triangle_width = 20  # Default triangle width
        self.triangle_height = 30  # Default triangle height
        self.gif_x_offset = 0  # Default gif x offset
        self.gif_y_offset = 0  # Default gif y offset
        self.initialize_tracker_type()

    def initialize_tracker_type(self):
        # Initialize the tracker type selection window
        self.type_window = tk.Tk()
        self.type_window.title("Select Tracker Type")
        
        # Tracker type selection
        ttk.Label(self.type_window, text="Select Tracker Type:").grid(row=0, column=0, padx=5, pady=5)
        self.tracker_type = tk.StringVar(value="triangle")
        ttk.Radiobutton(self.type_window, text="Triangle", variable=self.tracker_type, value="triangle").grid(row=1, column=0, padx=5, pady=5)
        ttk.Radiobutton(self.type_window, text="Image/GIF Overlay", variable=self.tracker_type, value="gif").grid(row=2, column=0, padx=5, pady=5)
        
        # Continue button
        ttk.Button(self.type_window, text="Continue", command=self.initialize_parameters).grid(row=3, column=0, pady=10)
        
        self.type_window.mainloop()

    def initialize_parameters(self):
        self.type_window.destroy()
        self.root = tk.Tk()
        self.root.title("Initialize Ball Tracker Parameters")

        # Common parameters
        row_counter = 0

        # Model Path
        ttk.Label(self.root, text="Model Path:").grid(row=row_counter, column=0, padx=5, pady=5)
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(self.root, textvariable=self.model_path_var, width=40)
        self.model_path_entry.grid(row=row_counter, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Select Model", command=self.select_model_path).grid(row=row_counter, column=2, padx=5, pady=5)
        row_counter += 1

        # Image Path
        ttk.Label(self.root, text="Image Path:").grid(row=row_counter, column=0, padx=5, pady=5)
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(self.root, textvariable=self.image_path_var, width=40)
        self.image_path_entry.grid(row=row_counter, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Select Image", command=self.select_image_path).grid(row=row_counter, column=2, padx=5, pady=5)
        row_counter += 1

        if self.tracker_type.get() == "triangle":
            # Triangle parameters
            ttk.Label(self.root, text="Triangle Width:").grid(row=row_counter, column=0, padx=5, pady=5)
            self.triangle_width_var = tk.StringVar(value="20")
            ttk.Entry(self.root, textvariable=self.triangle_width_var).grid(row=row_counter, column=1, padx=5, pady=5)
            row_counter += 1

            ttk.Label(self.root, text="Triangle Height:").grid(row=row_counter, column=0, padx=5, pady=5)
            self.triangle_height_var = tk.StringVar(value="30")
            ttk.Entry(self.root, textvariable=self.triangle_height_var).grid(row=row_counter, column=1, padx=5, pady=5)
        else:
            # GIF/Image parameters
            ttk.Label(self.root, text="Overlay File:").grid(row=row_counter, column=0, padx=5, pady=5)
            self.overlay_path_var = tk.StringVar()
            ttk.Entry(self.root, textvariable=self.overlay_path_var, width=40).grid(row=row_counter, column=1, padx=5, pady=5)
            ttk.Button(self.root, text="Select Overlay", command=self.select_overlay_path).grid(row=row_counter, column=2, padx=5, pady=5)
            row_counter += 1

            ttk.Label(self.root, text="X Offset:").grid(row=row_counter, column=0, padx=5, pady=5)
            self.gif_x_offset_var = tk.StringVar(value="0")
            ttk.Entry(self.root, textvariable=self.gif_x_offset_var).grid(row=row_counter, column=1, padx=5, pady=5)
            row_counter += 1

            ttk.Label(self.root, text="Y Offset:").grid(row=row_counter, column=0, padx=5, pady=5)
            self.gif_y_offset_var = tk.StringVar(value="0")
            ttk.Entry(self.root, textvariable=self.gif_y_offset_var).grid(row=row_counter, column=1, padx=5, pady=5)

        # Start Button
        ttk.Button(self.root, text="Start", command=self.start_app).grid(row=row_counter + 1, column=0, columnspan=3, pady=10)

        self.root.mainloop()

    def initialize_default_values(self):
        # Triangle defaults
        self.triangle_defaults = {
            "height": "30",
            "width": "20",
            "x_offset": "0",
            "y_offset": "15",
            "color": "0,0,255"
        }
        
        # Image overlay defaults
        self.image_defaults = {
            "height_factor": "0.014",
            "width_factor": "0.014",
            "x_offset": "0",
            "y_offset": "0",
            "overlay_path": ""
        }
        
        # GIF overlay defaults
        self.gif_defaults = {
            "height_factor": "0.06",
            "width_factor": "0.06",
            "x_offset": "0",
            "y_offset": "0",
            "overlay_path": ""
        }

    def setup_triangle_parameters(self, start_row):
        # Triangle Height
        ttk.Label(self.root, text="Triangle Height:").grid(row=start_row, column=0, padx=5, pady=5)
        self.height_var = tk.StringVar(value=self.triangle_defaults["height"])
        ttk.Entry(self.root, textvariable=self.height_var).grid(row=start_row, column=1, padx=5, pady=5)

        # Triangle Width
        ttk.Label(self.root, text="Triangle Width:").grid(row=start_row + 1, column=0, padx=5, pady=5)
        self.width_var = tk.StringVar(value=self.triangle_defaults["width"])
        ttk.Entry(self.root, textvariable=self.width_var).grid(row=start_row + 1, column=1, padx=5, pady=5)

        # X Offset
        ttk.Label(self.root, text="X Offset:").grid(row=start_row + 2, column=0, padx=5, pady=5)
        self.x_offset_var = tk.StringVar(value=self.triangle_defaults["x_offset"])
        ttk.Entry(self.root, textvariable=self.x_offset_var).grid(row=start_row + 2, column=1, padx=5, pady=5)

        # Y Offset
        ttk.Label(self.root, text="Y Offset:").grid(row=start_row + 3, column=0, padx=5, pady=5)
        self.y_offset_var = tk.StringVar(value=self.triangle_defaults["y_offset"])
        ttk.Entry(self.root, textvariable=self.y_offset_var).grid(row=start_row + 3, column=1, padx=5, pady=5)

        # Triangle Color
        ttk.Label(self.root, text="Triangle Color:").grid(row=start_row + 4, column=0, padx=5, pady=5)
        self.color_var = tk.StringVar(value=self.triangle_defaults["color"])
        self.color_button = ttk.Button(self.root, text="Pick Color", command=self.pick_color)
        self.color_button.grid(row=start_row + 4, column=1, padx=5, pady=5)

    def setup_image_parameters(self, start_row):
        # Height Factor
        ttk.Label(self.root, text="Height Factor:").grid(row=start_row, column=0, padx=5, pady=5)
        self.height_factor_var = tk.StringVar(value=self.image_defaults["height_factor"])
        ttk.Entry(self.root, textvariable=self.height_factor_var).grid(row=start_row, column=1, padx=5, pady=5)

        # Width Factor
        ttk.Label(self.root, text="Width Factor:").grid(row=start_row + 1, column=0, padx=5, pady=5)
        self.width_factor_var = tk.StringVar(value=self.image_defaults["width_factor"])
        ttk.Entry(self.root, textvariable=self.width_factor_var).grid(row=start_row + 1, column=1, padx=5, pady=5)

        # X Offset
        ttk.Label(self.root, text="X Offset:").grid(row=start_row + 2, column=0, padx=5, pady=5)
        self.x_offset_var = tk.StringVar(value=self.image_defaults["x_offset"])
        ttk.Entry(self.root, textvariable=self.x_offset_var).grid(row=start_row + 2, column=1, padx=5, pady=5)

        # Y Offset
        ttk.Label(self.root, text="Y Offset:").grid(row=start_row + 3, column=0, padx=5, pady=5)
        self.y_offset_var = tk.StringVar(value=self.image_defaults["y_offset"])
        ttk.Entry(self.root, textvariable=self.y_offset_var).grid(row=start_row + 3, column=1, padx=5, pady=5)

        # Overlay Image Path
        ttk.Label(self.root, text="Overlay Image:").grid(row=start_row + 4, column=0, padx=5, pady=5)
        self.overlay_path_var = tk.StringVar()
        ttk.Entry(self.root, textvariable=self.overlay_path_var, width=40).grid(row=start_row + 4, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Select Overlay", command=self.select_overlay_path).grid(row=start_row + 4, column=2, padx=5, pady=5)

    def setup_gif_parameters(self, start_row):
        # Similar to image parameters but for GIF
        self.setup_image_parameters(start_row)  # Reuse image parameter setup as they're the same

    def pick_color(self):
        color_code = askcolor(title="Pick a Color")[0]
        if color_code:
            self.color_var.set(f"{int(color_code[2])},{int(color_code[1])},{int(color_code[0])}")

    def select_model_path(self):
        selected_path = filedialog.askopenfilename(title="Select Model File")
        if selected_path:
            self.model_path_var.set(selected_path)

    def select_image_path(self):
        self.image_path_var.set(filedialog.askopenfilename(title="Select Image File"))

    def select_overlay_path(self):
        file_types = [('GIF/Image Files', '*.gif *.png *.jpg *.jpeg *.bmp')]
        selected_path = filedialog.askopenfilename(title="Select Overlay File", filetypes=file_types)
        if selected_path:
            self.overlay_path_var.set(selected_path)
            if selected_path.lower().endswith('.gif'):
                self.load_gif_frames(selected_path)
            else:
                self.load_image_as_gif(selected_path)

    def load_gif_frames(self, path):
        """Load GIF frames into memory"""
        gif = Image.open(path)
        self.gif_frames = []
        try:
            while True:
                frame = gif.convert('RGBA')
                self.gif_frames.append(np.array(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    def load_image_as_gif(self, path):
        """Load static image as single frame gif"""
        image = Image.open(path).convert('RGBA')
        self.gif_frames = [np.array(image)]


    def save_config(self):
        config = {
            "tracker_type": self.tracker_type.get(),
            "model_path": self.model_path_var.get(),
            "image_path": self.image_path_var.get(),
            "triangle": {
                "height": self.triangle_defaults["height"],
                "width": self.triangle_defaults["width"],
                "x_offset": self.triangle_defaults["x_offset"],
                "y_offset": self.triangle_defaults["y_offset"],
                "color": self.triangle_defaults["color"]
            },
            "image": {
                "height_factor": self.image_defaults["height_factor"],
                "width_factor": self.image_defaults["width_factor"],
                "x_offset": self.image_defaults["x_offset"],
                "y_offset": self.image_defaults["y_offset"],
                "overlay_path": self.image_defaults["overlay_path"]
            },
            "gif": {
                "height_factor": self.gif_defaults["height_factor"],
                "width_factor": self.gif_defaults["width_factor"],
                "x_offset": self.gif_defaults["x_offset"],
                "y_offset": self.gif_defaults["y_offset"],
                "overlay_path": self.gif_defaults["overlay_path"]
            }
        }

        # Update the active tracker type's values
        if self.tracker_type.get() == "triangle":
            config["triangle"] = {
                "height": self.height_var.get(),
                "width": self.width_var.get(),
                "x_offset": self.x_offset_var.get(),
                "y_offset": self.y_offset_var.get(),
                "color": self.color_var.get()
            }
        elif self.tracker_type.get() == "image":
            config["image"] = {
                "height_factor": self.height_factor_var.get(),
                "width_factor": self.width_factor_var.get(),
                "x_offset": self.x_offset_var.get(),
                "y_offset": self.y_offset_var.get(),
                "overlay_path": self.overlay_path_var.get()
            }
        else:  # gif
            config["gif"] = {
                "height_factor": self.height_factor_var.get(),
                "width_factor": self.width_factor_var.get(),
                "x_offset": self.x_offset_var.get(),
                "y_offset": self.y_offset_var.get(),
                "overlay_path": self.overlay_path_var.get()
            }

        with open("config.json", "w") as config_file:
            json.dump(config, config_file, indent=4)

        print("Configuration saved successfully!")

    def start_app(self):
        try:
            self.MODEL_PATH = self.model_path_var.get()
            self.IMAGE_PATH = self.image_path_var.get()
            
            # Update parameters based on tracker type
            if self.tracker_type.get() == "triangle":
                self.triangle_width = int(self.triangle_width_var.get())
                self.triangle_height = int(self.triangle_height_var.get())
            else:
                self.gif_x_offset = int(self.gif_x_offset_var.get())
                self.gif_y_offset = int(self.gif_y_offset_var.get())
            
            self.model = YOLO(self.MODEL_PATH)
            self.root.destroy()
            self.setup_gui()
        except Exception as e:
            print(f"Error initializing app: {e}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Ball Tracker")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack()

        # Process button
        ttk.Button(main_frame, text="Process Image", command=self.process_image).pack(pady=10)

        # Load and display initial image
        img = Image.open(self.IMAGE_PATH)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)

        self.root.mainloop()

    def display_image(self, image_path):
        # Load and display the full image
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

    def process_image(self):
        try:
            # Load and process the image
            frame = cv2.imread(self.IMAGE_PATH)
            result = self.model.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)
            ball_detections = detections[detections.class_id == 0]
            
            if len(ball_detections) > 0:
                coords = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                
                if self.tracker_type.get() == "triangle":
                    frame = self.draw_triangle_indicator(frame, coords)
                else:
                    frame = self.overlay_gif(frame, coords)

            # Update displayed image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            
        except Exception as e:
            print(f"Error processing image: {e}")

    def draw_triangle_indicator(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Draw a small filled inverted triangle above the ball, shifted upward"""
        if len(position) == 0:
            return frame
        try:
            # Calculate triangle vertices
            ball_x, ball_y = int(position[0]), int(position[1])
            
            # Parameters for triangle size and offset
            horizontal_offset = 3  # Shift triangle 3 pixels to the right
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

    def overlay_gif(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Overlay the GIF animation on the frame at the given position."""
        if len(position) == 0 or not self.gif_frames:
            return frame
        try:
            gif_frame = self.gif_frames[self.current_frame_idx]
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.gif_frames)
            # Adjust position with offset
            x = int(position[0] - gif_frame.shape[1] // 2 + self.gif_x_offset)
            y = int(position[1] - gif_frame.shape[0] // 2 + self.gif_y_offset)
            # Ensure the GIF fits within the frame
            h, w = gif_frame.shape[:2]
            y_start, y_end = max(0, y), min(y + h, frame.shape[0])
            x_start, x_end = max(0, x), min(x + w, frame.shape[1])
            gif_y_start, gif_y_end = max(0, -y), h - max(0, y + h - frame.shape[0])
            gif_x_start, gif_x_end = max(0, -x), w - max(0, x + w - frame.shape[1])
            # Convert GIF frame from RGB to BGR
            gif_rgb = gif_frame[gif_y_start:gif_y_end, gif_x_start:gif_x_end, :3]
            gif_bgr = cv2.cvtColor(gif_rgb, cv2.COLOR_RGB2BGR)
            # Extract alpha channel
            alpha_gif = gif_frame[gif_y_start:gif_y_end, gif_x_start:gif_x_end, 3] / 255.0
            alpha_frame = 1.0 - alpha_gif
            # Blend the GIF with the frame
            for c in range(3):  # BGR
                frame[y_start:y_end, x_start:x_end, c] = (
                    alpha_gif * gif_bgr[:, :, c] +
                    alpha_frame * frame[y_start:y_end, x_start:x_end, c]
                )
            return frame
        except Exception as e:
            print(f"Warning: Failed to overlay GIF: {str(e)}")
            return frame

if __name__ == "__main__":
    BallTrackerApp()