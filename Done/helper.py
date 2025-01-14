import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import json

TRIANGLE_COLOR = (0, 255,0)  # Green color for triangle
DEFAULT_GIF_SCALE = 0.06
DEFAULT_GIF_X_OFFSET = 0
DEFAULT_GIF_Y_OFFSET = 0


import tkinter as tk
from tkinter import ttk, colorchooser
import json

class LiveChangesWindow:
    def __init__(self, parent, tracker_type, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("Live Changes")
        self.tracker_type = tracker_type
        self.callback = callback
        
        # Create main frame
        self.frame = ttk.Frame(self.window)
        self.frame.pack(padx=10, pady=10)
        
        if tracker_type == "triangle":
            self.setup_triangle_controls()
        else:
            self.setup_overlay_controls()

    # Function to convert BGR to hex
    def bgr_to_hex(bgr_color):
        return '#{:02x}{:02x}{:02x}'.format(bgr_color[2], bgr_color[1], bgr_color[0])


    def setup_triangle_controls(self):
        # Triangle controls
        ttk.Label(self.frame, text="Triangle Height:").pack()
        self.height_var = tk.StringVar(value="30")
        ttk.Entry(self.frame, textvariable=self.height_var).pack()

        ttk.Label(self.frame, text="Triangle Width:").pack()
        self.width_var = tk.StringVar(value="20")
        ttk.Entry(self.frame, textvariable=self.width_var).pack()

        ttk.Label(self.frame, text="X Offset:").pack()
        self.x_offset_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.x_offset_var).pack()

        ttk.Label(self.frame, text="Y Offset:").pack()
        self.y_offset_var = tk.StringVar(value="15")
        ttk.Entry(self.frame, textvariable=self.y_offset_var).pack()

        # Hex color input
        ttk.Label(self.frame, text="Triangle Color (Hex):").pack(pady=5)
        self.hex_color_var = tk.StringVar() 
        self.hex_color_entry = ttk.Entry(self.frame, textvariable=self.hex_color_var)
        self.hex_color_entry.pack(pady=5)

        # Apply Color Button
        ttk.Button(self.frame, text="Apply Color", command=self.apply_color).pack(pady=5)

        # Bind changes to update callback
        for var in [self.height_var, self.width_var, self.x_offset_var, self.y_offset_var]:
            var.trace_add("write", self.on_value_change)

    # Function to convert hex to BGR
    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')  # Remove '#' if present
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]  # Convert to RGB
        return (rgb[2], rgb[1], rgb[0])  # Return in BGR format for OpenCV
    
    def apply_color(self):
        # Get the hex color from the input field
        hex_color = self.hex_color_var.get()

        # Convert hex to BGR
        bgr_color = self.hex_to_bgr(hex_color)

        # Update the global TRIANGLE_COLOR variable
        global TRIANGLE_COLOR
        TRIANGLE_COLOR = bgr_color

        self.callback({
            "height": int(self.height_var.get()),
            "width": int(self.width_var.get()),
            "x_offset": int(self.x_offset_var.get()),
            "y_offset": int(self.y_offset_var.get()),
            "color": TRIANGLE_COLOR
        })

    def setup_overlay_controls(self):
        # Overlay controls (unchanged)
        ttk.Label(self.frame, text="Scale Factor:").pack()
        self.scale_var = tk.StringVar(value="1.0")
        ttk.Entry(self.frame, textvariable=self.scale_var).pack()
        
        ttk.Label(self.frame, text="X Offset:").pack()
        self.x_offset_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.x_offset_var).pack()
        
        ttk.Label(self.frame, text="Y Offset:").pack()
        self.y_offset_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.y_offset_var).pack()

        # Save Configuration Button
        ttk.Button(self.frame, text="Save Configuration", command=self.save_config).pack(pady=10)
        
        # Bind changes to update callback
        for var in [self.scale_var, self.x_offset_var, self.y_offset_var]:
            var.trace_add("write", self.on_value_change)

    def on_value_change(self, *args):
        try:
            values = {}
            if self.tracker_type == "triangle":
                values = {
                    "height": int(self.height_var.get()),
                    "width": int(self.width_var.get()),
                    "x_offset": int(self.x_offset_var.get()),
                    "y_offset": int(self.y_offset_var.get()),
                    "color": TRIANGLE_COLOR
                }
            else:
                values = {
                    "scale": float(self.scale_var.get()),
                    "x_offset": int(self.x_offset_var.get()),
                    "y_offset": int(self.y_offset_var.get())
                }
            self.callback(values)
        except ValueError:
            pass  # Ignore invalid values

    # Function to convert BGR color to the desired string format
    def bgr_to_string(self, bgr_color):
        return f"({bgr_color[0]},{bgr_color[1]},{bgr_color[2]})"

    def get_config(self):
        # Convert TRIANGLE_COLOR to the desired string format
        triangle_color_str = self.bgr_to_string(TRIANGLE_COLOR)
        
        config = {
            "triangle": {
                "height": self.height_var.get() if self.tracker_type == "triangle" else "30",
                "width": self.width_var.get() if self.tracker_type == "triangle" else "20",
                "x_offset": self.x_offset_var.get() if self.tracker_type == "triangle" else "0",
                "y_offset": self.y_offset_var.get() if self.tracker_type == "triangle" else "15",
                "color": triangle_color_str if self.tracker_type == "triangle" else "(0, 255, 0)"
            },
            "overlay": {
                "scale": self.scale_var.get() if self.tracker_type == "gif" else "0.06",
                "x_offset": self.x_offset_var.get() if self.tracker_type == "gif" else "0",
                "y_offset": self.y_offset_var.get() if self.tracker_type == "gif" else "0"
            }
        }
        return config


class BallTrackerApp:
    def __init__(self):
        # Initialize previous variables...
        self.gif_frames = []
        self.current_frame_idx = 0
        self.triangle_width = 20
        self.triangle_height = 30
        self.triangle_x_offset = 0
        self.triangle_y_offset = 15
        self.gif_x_offset = 0
        self.gif_y_offset = 0
        self.gif_scale = 1.0
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
            if selected_path.lower().endswith('.gif') or selected_path.lower().endswith('.webp'):
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
            self.save_config()
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

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        # Process button
        ttk.Button(button_frame, text="Process Image", command=self.process_image).pack(side=tk.LEFT, padx=5)
        
        # Live Changes button
        ttk.Button(button_frame, text="Live Changes", command=self.open_live_changes).pack(side=tk.LEFT, padx=5)

        # Load and display initial image
        img = Image.open(self.IMAGE_PATH)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)

        self.root.mainloop()

    def open_live_changes(self):
        self.live_changes = LiveChangesWindow(self.root, self.tracker_type.get(), self.update_parameters)

    def update_parameters(self, values):
        if self.tracker_type.get() == "triangle":
            print("here")
            self.triangle_height = values["height"]
            self.triangle_width = values["width"]
            self.triangle_x_offset = values["x_offset"]
            self.triangle_y_offset = values["y_offset"]
        else:
            self.gif_scale = values["scale"]
            self.gif_x_offset = values["x_offset"]
            self.gif_y_offset = values["y_offset"]
        
        # Reprocess the image with new parameters
        self.process_image()

    def save_config(self):
        config = self.live_changes.get_config() if hasattr(self, 'live_changes') else {
            "triangle": {
                "height": "30",
                "width": "20",
                "x_offset": "0",
                "y_offset": "15",
                "color": "(0, 255, 0)"
            },
            "overlay": {
                "scale": "0.06",
                "x_offset": "1.0",
                "y_offset": "1.0"
            }
        }
        try:
            with open("tracker_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print("Configuration saved successfully!")
        except Exception as e:
            print(f"Failed to save configuration: {str(e)}")

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
            self.save_config()
        except Exception as e:
            print(f"Error processing image: {e}")

    def draw_triangle_indicator(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        if len(position) == 0:
            return frame
        try:
            ball_x, ball_y = int(position[0]), int(position[1])
            
            # Use the live parameters
            horizontal_offset = self.triangle_x_offset
            offset = self.triangle_y_offset
            triangle_width = self.triangle_width
            triangle_height = self.triangle_height // 2
            
            top_point = (ball_x + horizontal_offset, ball_y - offset)
            left_point = (ball_x - triangle_width // 2 + horizontal_offset, ball_y - triangle_height - offset)
            right_point = (ball_x + triangle_width // 2 + horizontal_offset, ball_y - triangle_height - offset)
            
            print(f"Triangle points: {top_point}, {left_point}, {right_point}")


            pts = np.array([top_point, left_point, right_point], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], TRIANGLE_COLOR, cv2.LINE_AA)
            
            return frame
        except Exception as e:
            print(f"Warning: Failed to draw filled inverted triangle: {str(e)}")
            return frame

    def overlay_gif(self, frame: np.ndarray, position: np.ndarray) -> np.ndarray:
        if len(position) == 0 or not self.gif_frames:
            return frame
        try:
            gif_frame = self.gif_frames[self.current_frame_idx]
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.gif_frames)
            
            # Scale the gif frame
            h, w = gif_frame.shape[:2]
            new_h, new_w = int(h * self.gif_scale), int(w * self.gif_scale)
            gif_frame = cv2.resize(gif_frame, (new_w, new_h))
            
            # Apply offsets
            x = int(position[0] - new_w // 2 + self.gif_x_offset)
            y = int(position[1] - new_h // 2 + self.gif_y_offset)
            
            # Rest of the overlay code remains the same...
            h, w = gif_frame.shape[:2]
            y_start, y_end = max(0, y), min(y + h, frame.shape[0])
            x_start, x_end = max(0, x), min(x + w, frame.shape[1])
            gif_y_start, gif_y_end = max(0, -y), h - max(0, y + h - frame.shape[0])
            gif_x_start, gif_x_end = max(0, -x), w - max(0, x + w - frame.shape[1])
            
            gif_rgb = gif_frame[gif_y_start:gif_y_end, gif_x_start:gif_x_end, :3]
            gif_bgr = cv2.cvtColor(gif_rgb, cv2.COLOR_RGB2BGR)
            alpha_gif = gif_frame[gif_y_start:gif_y_end, gif_x_start:gif_x_end, 3] / 255.0
            alpha_frame = 1.0 - alpha_gif
            
            for c in range(3):
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