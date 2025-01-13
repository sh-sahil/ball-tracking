import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import json

class BallTrackerApp:
    def __init__(self):
        self.initialize_tracker_type()

    def initialize_tracker_type(self):
        # Initialize the tracker type selection window
        self.type_window = tk.Tk()
        self.type_window.title("Select Tracker Type")
        
        # Tracker type selection
        ttk.Label(self.type_window, text="Select Tracker Type:").grid(row=0, column=0, padx=5, pady=5)
        self.tracker_type = tk.StringVar(value="triangle")
        ttk.Radiobutton(self.type_window, text="Triangle", variable=self.tracker_type, value="triangle").grid(row=1, column=0, padx=5, pady=5)
        ttk.Radiobutton(self.type_window, text="Image Overlay", variable=self.tracker_type, value="image").grid(row=2, column=0, padx=5, pady=5)
        ttk.Radiobutton(self.type_window, text="GIF Overlay", variable=self.tracker_type, value="gif").grid(row=3, column=0, padx=5, pady=5)
        
        # Continue button
        ttk.Button(self.type_window, text="Continue", command=self.initialize_parameters).grid(row=4, column=0, pady=10)
        
        self.type_window.mainloop()

    def initialize_parameters(self):
        self.type_window.destroy()
        self.root = tk.Tk()
        self.root.title("Initialize Ball Tracker Parameters")

        # Common parameters for all types
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

        # Initialize default values for all types
        self.initialize_default_values()

        if self.tracker_type.get() == "triangle":
            self.setup_triangle_parameters(row_counter)
        elif self.tracker_type.get() == "image":
            self.setup_image_parameters(row_counter)
        else:  # gif
            self.setup_gif_parameters(row_counter)

        # Start Button
        ttk.Button(self.root, text="Start", command=self.start_app).grid(row=row_counter + 6, column=0, columnspan=3, pady=10)

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
        file_types = [('All Files', '*.*')]
        if self.tracker_type.get() == "gif":
            file_types = [('GIF Files', '*.gif *webp')]
        elif self.tracker_type.get() == "image":
            file_types = [('Image Files', '*.png *.jpg *.jpeg *.bmp')]
            
        selected_path = filedialog.askopenfilename(title="Select Overlay File", filetypes=file_types)
        if selected_path:
            self.overlay_path_var.set(selected_path)

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
            self.model = YOLO(self.MODEL_PATH)
            self.save_config()
            self.root.destroy()
            self.setup_gui()
        except Exception as e:
            print(f"Error initializing app: {e}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Ball Tracker Parameter Tuner")

        # Image display window
        self.image_window = tk.Toplevel(self.root)
        self.image_window.title("Full Image Display")

        # Scrollable frame for full image
        self.scroll_canvas = tk.Canvas(self.image_window)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.image_window, orient="vertical", command=self.scroll_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.bind('<Configure>', lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.image_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()

        # Display the selected image
        self.display_image(self.IMAGE_PATH)

        # Parameter adjustment window
        controls_frame = ttk.LabelFrame(self.root, text=f"Adjust {self.tracker_type.get().title()} Parameters", padding="10")
        controls_frame.grid(row=0, column=0, padx=10, pady=10)

        row = 0
        if self.tracker_type.get() == "triangle":
            # Triangle Height
            ttk.Label(controls_frame, text="Triangle Height:").grid(row=row, column=0, padx=5)
            self.height_var = tk.StringVar(value=self.triangle_defaults["height"])
            ttk.Entry(controls_frame, textvariable=self.height_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Triangle Width
            ttk.Label(controls_frame, text="Triangle Width:").grid(row=row, column=0, padx=5)
            self.width_var = tk.StringVar(value=self.triangle_defaults["width"])
            ttk.Entry(controls_frame, textvariable=self.width_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # X Offset
            ttk.Label(controls_frame, text="X Offset:").grid(row=row, column=0, padx=5)
            self.x_offset_var = tk.StringVar(value=self.triangle_defaults["x_offset"])
            ttk.Entry(controls_frame, textvariable=self.x_offset_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Y Offset
            ttk.Label(controls_frame, text="Y Offset:").grid(row=row, column=0, padx=5)
            self.y_offset_var = tk.StringVar(value=self.triangle_defaults["y_offset"])
            ttk.Entry(controls_frame, textvariable=self.y_offset_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Triangle Color
            ttk.Label(controls_frame, text="Triangle Color:").grid(row=row, column=0, padx=5)
            self.color_var = tk.StringVar(value=self.triangle_defaults["color"])
            ttk.Button(controls_frame, text="Pick Color", command=self.pick_color).grid(row=row, column=1, padx=5)

        else:  # image or gif
            defaults = self.image_defaults if self.tracker_type.get() == "image" else self.gif_defaults

            # Height Factor
            ttk.Label(controls_frame, text="Height Factor:").grid(row=row, column=0, padx=5)
            self.height_factor_var = tk.StringVar(value=defaults["height_factor"])
            ttk.Entry(controls_frame, textvariable=self.height_factor_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Width Factor
            ttk.Label(controls_frame, text="Width Factor:").grid(row=row, column=0, padx=5)
            self.width_factor_var = tk.StringVar(value=defaults["width_factor"])
            ttk.Entry(controls_frame, textvariable=self.width_factor_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # X Offset
            ttk.Label(controls_frame, text="X Offset:").grid(row=row, column=0, padx=5)
            self.x_offset_var = tk.StringVar(value=defaults["x_offset"])
            ttk.Entry(controls_frame, textvariable=self.x_offset_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Y Offset
            ttk.Label(controls_frame, text="Y Offset:").grid(row=row, column=0, padx=5)
            self.y_offset_var = tk.StringVar(value=defaults["y_offset"])
            ttk.Entry(controls_frame, textvariable=self.y_offset_var, width=10).grid(row=row, column=1, padx=5)
            row += 1

            # Overlay Path
            ttk.Label(controls_frame, text=f"Overlay {self.tracker_type.get().title()}:").grid(row=row, column=0, padx=5)
            self.overlay_path_var = tk.StringVar(value=defaults["overlay_path"])
            ttk.Entry(controls_frame, textvariable=self.overlay_path_var, width=30).grid(row=row, column=1, padx=5)
            ttk.Button(controls_frame, text="Browse", command=self.select_overlay_path).grid(row=row, column=2, padx=5)

        # Process button at the bottom
        ttk.Button(controls_frame, text="Apply Changes", command=self.process_image).grid(row=row + 1, column=0, columnspan=3, pady=10)

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
                    height = int(self.height_var.get())
                    width = int(self.width_var.get())
                    x_offset = int(self.x_offset_var.get())
                    y_offset = int(self.y_offset_var.get())
                    color = tuple(map(int, self.color_var.get().split(',')))
                    frame = self.draw_triangle_indicator(frame, coords, height, width, x_offset, y_offset, color)
                else:  # image or gif
                    height_factor = float(self.height_factor_var.get())
                    width_factor = float(self.width_factor_var.get())
                    x_offset = int(self.x_offset_var.get())
                    y_offset = int(self.y_offset_var.get())
                    overlay_path = self.overlay_path_var.get()
                    if self.tracker_type.get() == "image":
                        frame = self.draw_image_overlay(frame, coords, height_factor, width_factor, x_offset, y_offset, overlay_path)
                    else:
                        frame = self.draw_gif_overlay(frame, coords, height_factor, width_factor, x_offset, y_offset, overlay_path)

            # Update displayed image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.save_config()
        except Exception as e:
            print(f"Error processing image: {e}")

    def draw_triangle_indicator(self, frame, position, height, width, x_offset, y_offset, color):
        try:
            ball_x, ball_y = int(position[0]), int(position[1])
            top_point = (ball_x + x_offset, ball_y - y_offset)
            left_point = (ball_x + x_offset - width // 2, ball_y - height - y_offset)
            right_point = (ball_x + x_offset + width // 2, ball_y - height - y_offset)

            pts = np.array([top_point, left_point, right_point], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], color)
            return frame
        except Exception as e:
            print(f"Error drawing triangle: {e}")
            return frame

    def draw_image_overlay(self, frame, position, height_factor, width_factor, x_offset, y_offset, overlay_path):
        try:
            # Read image with alpha channel
            overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
            if overlay is None:
                print("Error: Could not load overlay image")
                return frame
                
            # Calculate new dimensions
            new_height = int(overlay.shape[0] * height_factor)
            new_width = int(overlay.shape[1] * width_factor)
            overlay = cv2.resize(overlay, (new_width, new_height))
            
            # Calculate position
            ball_x, ball_y = int(position[0]), int(position[1])
            x = ball_x + x_offset - new_width // 2
            y = ball_y + y_offset - new_height
            
            # Ensure coordinates are within frame bounds
            if x < 0: x = 0
            if y < 0: y = 0
            if x + new_width > frame.shape[1]: new_width = frame.shape[1] - x
            if y + new_height > frame.shape[0]: new_height = frame.shape[0] - y
            
            # Check if overlay has alpha channel
            if overlay.shape[2] == 4:
                # Separate RGB and alpha channels
                overlay_rgb = overlay[:new_height, :new_width, :3]
                alpha = overlay[:new_height, :new_width, 3] / 255.0
                alpha = np.dstack([alpha] * 3)  # Make alpha same shape as RGB
                
                # Get the region of interest from the frame
                roi = frame[y:y + new_height, x:x + new_width]
                
                # Blend the overlay with the frame using alpha channel
                blended = cv2.addWeighted(
                    overlay_rgb.astype(float),
                    1,
                    roi.astype(float),
                    0,
                    0
                )
                frame[y:y + new_height, x:x + new_width] = (alpha * blended + (1 - alpha) * roi).astype(np.uint8)
            else:
                # If no alpha channel, just overlay RGB
                frame[y:y + new_height, x:x + new_width] = overlay[:new_height, :new_width, :]
            
            return frame
        except Exception as e:
            print(f"Error drawing image overlay: {e}")
            return frame

    def draw_gif_overlay(self, frame, position, height_factor, width_factor, x_offset, y_offset, overlay_path):
        try:
            # Open and convert GIF
            gif = Image.open(overlay_path)
            gif = gif.convert('RGBA')
            overlay = np.array(gif)
            
            # Calculate new dimensions
            new_height = int(overlay.shape[0] * height_factor)
            new_width = int(overlay.shape[1] * width_factor)
            overlay = cv2.resize(overlay, (new_width, new_height))
            
            # Calculate position
            ball_x, ball_y = int(position[0]), int(position[1])
            x = ball_x + x_offset - new_width // 2
            y = ball_y + y_offset - new_height
            
            # Ensure coordinates are within frame bounds
            if x < 0: x = 0
            if y < 0: y = 0
            if x + new_width > frame.shape[1]: new_width = frame.shape[1] - x
            if y + new_height > frame.shape[0]: new_height = frame.shape[0] - y
            
            # Convert RGBA to BGR format and handle alpha channel
            overlay_rgb = cv2.cvtColor(overlay[:new_height, :new_width], cv2.COLOR_RGBA2BGR)
            alpha = overlay[:new_height, :new_width, 3] / 255.0
            alpha = np.dstack([alpha] * 3)  # Make alpha same shape as RGB
            
            # Get the region of interest from the frame
            roi = frame[y:y + new_height, x:x + new_width]
            
            # Blend the overlay with the frame using alpha channel
            blended = cv2.addWeighted(
                overlay_rgb.astype(float),
                1,
                roi.astype(float),
                0,
                0
            )
            frame[y:y + new_height, x:x + new_width] = (alpha * blended + (1 - alpha) * roi).astype(np.uint8)
            
            return frame
        except Exception as e:
            print(f"Error drawing GIF overlay: {e}")
            return frame

if __name__ == "__main__":
    BallTrackerApp()
