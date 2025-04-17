import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class ShelfDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shelf Detection")
        self.canvas = None
        self.image = None
        self.original_image = None
        self.tk_img = None

        self.top_line = None
        self.bottom_line = None
        self.left_bound = None
        self.right_bound = None

        self.rect_start = None
        self.draw_rect_mode = False

        self.max_width = 800
        self.max_height = 600

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        self.process_button = tk.Button(self.root, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.count_label = tk.Label(self.root, text="Number of gaps: 0")
        self.count_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.image = self.resize_image(self.original_image)
            self.show_image()

    def resize_image(self, image):
        height, width = image.shape[:2]
        scale_width = self.max_width / width
        scale_height = self.max_height / height
        scale = min(scale_width, scale_height)
        self.scale_ratio = scale

        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def show_image(self):
        if self.canvas:
            self.canvas.destroy()

        self.canvas = tk.Canvas(self.canvas_frame, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.pack()

        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        return int(canvas_x / self.scale_ratio), int(canvas_y / self.scale_ratio)

    def on_mouse_press(self, event):
        self.rect_start = (event.x, event.y)
        self.draw_rect_mode = True

    def on_mouse_drag(self, event):
        if self.rect_start:
            self.canvas.delete("rect")
            self.canvas.create_rectangle(
                self.rect_start[0], self.rect_start[1], event.x, event.y, outline="red", width=2, tags="rect"
            )

    def on_mouse_release(self, event):
        if self.rect_start:
            x1, y1 = self.rect_start
            x2, y2 = event.x, event.y

            # Convert canvas coords to image coords
            img_x1, img_y1 = self.canvas_to_image_coords(x1, y1)
            img_x2, img_y2 = self.canvas_to_image_coords(x2, y2)

            self.left_bound = min(img_x1, img_x2)
            self.right_bound = max(img_x1, img_x2)
            self.top_line = min(img_y1, img_y2)
            self.bottom_line = max(img_y1, img_y2)

            self.draw_rect_mode = False
            self.process_image()

    def process_image(self):
        if None in [self.top_line, self.bottom_line, self.left_bound, self.right_bound]:
            messagebox.showerror("Error", "Please draw a full rectangle to select shelf area!")
            return

        self.detect_shelves_manually()

    def detect_shelves_manually(self):
        image = self.original_image
        roi = image[self.top_line:self.bottom_line, self.left_bound:self.right_bound].copy()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        custom_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        filtered = cv2.filter2D(blur, -1, custom_kernel)

        edges = cv2.Canny(filtered, 50, 150)
        kernel = np.ones((1, 5), np.uint8)
        edges_morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_clean = cv2.GaussianBlur(edges_morph, (3, 3), 0)

        lines = cv2.HoughLinesP(
            edges_clean, rho=2, theta=np.pi / 180,
            threshold=100, minLineLength=200, maxLineGap=5
        )

        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if 75 <= abs(angle) <= 105:
                    vertical_lines.append(((x1, y1), (x2, y2)))

        grouped_xs = self.group_close_lines(vertical_lines, distance_threshold=20)
        grouped_xs.sort()

        for x in grouped_xs:
            cv2.line(roi, (x, 0), (x, roi.shape[0]), (0, 0, 255), 2)

        total_gaps = 0
        for j in range(len(grouped_xs) - 1):
            x_start = grouped_xs[j]
            x_end = grouped_xs[j + 1]
            gap = x_end - x_start
            x_mid = (x_start + x_end) // 2
            y_text = roi.shape[0] // 2

            # Label gap number and width
            label = f"{total_gaps + 1}"
            cv2.putText(
                roi, label, (x_mid - 40, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (57, 255, 20), 2
            )

            total_gaps += 1

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 3))
        plt.imshow(roi_rgb)
        plt.title("Selected Shelf Region with Detected Gaps")
        plt.axis("off")
        plt.show()

        self.count_label.config(text=f"Number of gaps: {total_gaps}")

    def group_close_lines(self, lines, distance_threshold=20):
        if not lines:
            return []

        x_positions = [(x1 + x2) // 2 for ((x1, y1), (x2, y2)) in lines]
        x_positions.sort()

        grouped = []
        group = [x_positions[0]]

        for x in x_positions[1:]:
            if abs(x - group[-1]) <= distance_threshold:
                group.append(x)
            else:
                grouped.append(group)
                group = [x]
        grouped.append(group)

        representative_lines = [int(np.mean(g)) for g in grouped]
        return representative_lines


if __name__ == "__main__":
    root = tk.Tk()
    app = ShelfDetectorApp(root)
    root.mainloop()
