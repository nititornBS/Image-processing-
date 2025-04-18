# 📚 Bookshelf Gap Detection with OpenCV

This project detects vertical gaps between books on each shelf of a bookshelf using computer vision techniques. It helps visualize spacing between books and can be useful for automated shelf organization or analysis.

## 📸 Example Use Case

Given a bookshelf image, the system:
1. Detects shelf regions manually (predefined lines).
2. Applies image processing techniques to detect vertical lines (gaps).
3. Groups nearby lines and displays the gap distances.
4. Visualizes results with gap counts on each shelf.

## 🧰 Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

## ⚙️ Method Overview

1. **Manual Shelf Detection**  
   Predefined Y-coordinate lines mark the top and bottom of each shelf.

2. **Preprocessing**
   - Convert shelf image to grayscale.
   - Apply **Bilateral Filter** (low-pass filter) for noise reduction while keeping edges.
   - Use **custom edge detection kernel** and **Canny Edge Detector**.

3. **Morphological Processing**
   - Apply morphological closing to bridge broken edges.
   - Gaussian Blur (low-pass filter) for smoothing.

4. **Line Detection**
   - Use `cv2.HoughLinesP` to detect vertical lines (gaps between books).
   - Group close lines to avoid duplicate gap detections.

5. **Gap Visualization**
   - Draw vertical lines for gaps.
   - Display distance between each pair of detected gaps on the shelf.

## 📂 File Structure

```bash
.
├── image/
│   └── 4.jpg              # Example bookshelf image
├── detect_shelves.py      # Main detection script
└── README.md
