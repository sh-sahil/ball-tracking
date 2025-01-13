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
        self.initialize_parameters()
        # self.setup_main_window()

    def initialize_parameters(self):
        # Initialize the parameter window
        self.root = tk.Tk()
        self.root.title("Initialize Ball Tracker Parameters")

        # Triangle Height
        ttk.Label(self.root, text="Triangle Height:").grid(row=0, column=0, padx=5, pady=5)
        self.height_var = tk.StringVar(value="30")
        ttk.Entry(self.root, textvariable=self.height_var).grid(row=0, column=1, padx=5, pady=5)

        # Triangle Width
        ttk.Label(self.root, text="Triangle Width:").grid(row=1, column=0, padx=5, pady=5)
        self.width_var = tk.StringVar(value="20")
        ttk.Entry(self.root, textvariable=self.width_var).grid(row=1, column=1, padx=5, pady=5)

        # Vertical Offset
        ttk.Label(self.root, text="Vertical Offset:").grid(row=2, column=0, padx=5, pady=5)
        self.offset_var = tk.StringVar(value="15")
        ttk.Entry(self.root, textvariable=self.offset_var).grid(row=2, column=1, padx=5, pady=5)

        # Triangle Color Picker
        ttk.Label(self.root, text="Triangle Color:").grid(row=3, column=0, padx=5, pady=5)
        self.color_var = tk.StringVar(value="0,0,255")
        self.color_button = ttk.Button(self.root, text="Pick Color", command=self.pick_color)
        self.color_button.grid(row=3, column=1, padx=5, pady=5)

        # Model Path
        ttk.Label(self.root, text="Model Path:").grid(row=4, column=0, padx=5, pady=5)
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(self.root, textvariable=self.model_path_var, width=40)
        self.model_path_entry.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Select Model", command=self.select_model_path).grid(row=4, column=2, padx=5, pady=5)

        # Image Path
        ttk.Label(self.root, text="Image Path:").grid(row=5, column=0, padx=5, pady=5)
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(self.root, textvariable=self.image_path_var, width=40)
        self.image_path_entry.grid(row=5, column=1, padx=5, pady=5)
        ttk.Button(self.root, text="Select Image", command=self.select_image_path).grid(row=5, column=2, padx=5, pady=5)

        # Start Button
        ttk.Button(self.root, text="Start", command=self.start_app).grid(row=6, column=0, columnspan=3, pady=10)

        self.root.mainloop()

    def pick_color(self):
        # Open color picker and update color variable
        color_code = askcolor(title="Pick a Color")[0]
        if color_code:
            self.color_var.set(f"{int(color_code[2])},{int(color_code[1])},{int(color_code[0])}")

    def select_model_path(self):
        # Open file dialog to select the model path
        selected_path = filedialog.askopenfilename(title="Select Model File")
        if selected_path:
            self.model_path_var.set(selected_path)

    def select_image_path(self):
        # Open file dialog to select the image path
        self.image_path_var.set(filedialog.askopenfilename(title="Select Image File"))

    def start_app(self):
        try:
            self.DEFAULT_HEIGHT = int(self.height_var.get())
            self.DEFAULT_WIDTH = int(self.width_var.get())
            self.DEFAULT_OFFSET = int(self.offset_var.get())
            self.TRIANGLE_COLOR = tuple(map(int, self.color_var.get().split(',')))
            self.MODEL_PATH = self.model_path_var.get()
            self.IMAGE_PATH = self.image_path_var.get()

            self.model = YOLO(self.MODEL_PATH)
            self.save_config()
            self.root.destroy()
            self.setup_gui()
        except Exception as e:
            print(f"Error initializing app: {e}")

    def save_config(self):
            height = int(self.height_var.get())
            width = int(self.width_var.get())
            offset = int(self.offset_var.get())
            color = tuple(map(int, self.color_var.get().split(',')))
            config = {
                

                "triangle_height": height,
                "triangle_width": width,
                "vertical_offset": offset,
                "triangle_color": color,
                "model_path": self.MODEL_PATH,
                "image_path": self.IMAGE_PATH
            }
            with open("config.json", "w") as config_file:
                json.dump(config, config_file, indent=4)

            print("Configuration saved successfully!")

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
        controls_frame = ttk.LabelFrame(self.root, text="Adjust Parameters", padding="10")
        controls_frame.grid(row=0, column=0, padx=10, pady=10)

        # Triangle Height
        ttk.Label(controls_frame, text="Triangle Height:").grid(row=0, column=0, padx=5)
        self.height_var = tk.StringVar(value=str(self.DEFAULT_HEIGHT))
        self.height_entry = ttk.Entry(controls_frame, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=0, column=1, padx=5)

        # Triangle Width
        ttk.Label(controls_frame, text="Triangle Width:").grid(row=1, column=0, padx=5)
        self.width_var = tk.StringVar(value=str(self.DEFAULT_WIDTH))
        self.width_entry = ttk.Entry(controls_frame, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=1, column=1, padx=5)

        # Vertical Offset
        ttk.Label(controls_frame, text="Vertical Offset:").grid(row=2, column=0, padx=5)
        self.offset_var = tk.StringVar(value=str(self.DEFAULT_OFFSET))
        self.offset_entry = ttk.Entry(controls_frame, textvariable=self.offset_var, width=10)
        self.offset_entry.grid(row=2, column=1, padx=5)

        # Triangle Color
        ttk.Label(controls_frame, text="Triangle Color (B,G,R):").grid(row=3, column=0, padx=5)
        self.color_var = tk.StringVar(value=",".join(map(str, self.TRIANGLE_COLOR)))
        self.color_entry = ttk.Entry(controls_frame, textvariable=self.color_var, width=15)
        self.color_entry.grid(row=3, column=1, padx=5)

        # Process button
        ttk.Button(controls_frame, text="Apply Changes", command=self.process_image).grid(row=4, column=0, columnspan=2, pady=10)
        self.save_config()

        self.root.mainloop()

    def display_image(self, image_path):
        # Load and display the full image
        image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

    def process_image(self):
        try:
            height = int(self.height_var.get())
            width = int(self.width_var.get())
            offset = int(self.offset_var.get())
            color = tuple(map(int, self.color_var.get().split(',')))

            # Load and process the image
            frame = cv2.imread(self.IMAGE_PATH)
            result = self.model.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)
            ball_detections = detections[detections.class_id == 0]
            self.save_config()

            if len(ball_detections) > 0:
                coords = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                frame = self.draw_triangle_indicator(frame, coords, height, width, offset, color)

            # Update displayed image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
        except Exception as e:
            print(f"Error processing image: {e}")

    def draw_triangle_indicator(self, frame, position, height, width, offset, color):
        try:
            ball_x, ball_y = int(position[0]), int(position[1])
            top_point = (ball_x, ball_y - offset)
            left_point = (ball_x - width // 2, ball_y - height - offset)
            right_point = (ball_x + width // 2, ball_y - height - offset)

            pts = np.array([top_point, left_point, right_point], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(frame, [pts], color)
            return frame
        except Exception as e:
            print(f"Error drawing triangle: {e}")
            return frame

if __name__ == "__main__":
    BallTrackerApp()
