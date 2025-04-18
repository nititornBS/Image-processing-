import cv2
import numpy as np
import matplotlib.pyplot as plt

def group_close_lines(lines, distance_threshold=20):
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

def detect_shelves_manually(image_path, manual_lines):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result_manual = image.copy()
    for line in manual_lines:
        cv2.line(result_manual, (0, line), (result_manual.shape[1], line), (0, 255, 0), 2)

    manual_lines = sorted(manual_lines)

    for i in range(len(manual_lines) - 1):
        y1, y2 = manual_lines[i], manual_lines[i + 1]
        shelf_region = image[y1:y2, :]

        shelf_gray = cv2.cvtColor(shelf_region, cv2.COLOR_BGR2GRAY)
        shelf_blur = cv2.bilateralFilter(shelf_gray, 9, 75, 75)

        custom_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        shelf_filtered = cv2.filter2D(shelf_blur, -1, custom_kernel)

        shelf_edges = cv2.Canny(shelf_filtered, 50, 150)
        kernel = np.ones((1, 5), np.uint8)
        shelf_edges_morph = cv2.morphologyEx(shelf_edges, cv2.MORPH_CLOSE, kernel)
        shelf_edges_clean = cv2.GaussianBlur(shelf_edges_morph, (3, 3), 0)

        plt.figure(figsize=(10, 3))
        plt.imshow(shelf_edges_clean, cmap='gray')
        plt.title(f"Shelf {i+1} Edges After Morphology + Lowpass")
        plt.axis("off")

        lines = cv2.HoughLinesP(
            shelf_edges_clean,
            rho=2,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=200,
            maxLineGap=5
        )

        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1_l, x2, y2_l = line[0]
                angle = np.arctan2(y2_l - y1_l, x2 - x1) * 180.0 / np.pi
                if 75 <= abs(angle) <= 105:
                    vertical_lines.append(((x1, y1_l), (x2, y2_l)))

        grouped_xs = group_close_lines(vertical_lines, distance_threshold=20)
        grouped_xs.sort()

        for idx, x in enumerate(grouped_xs):
            cv2.line(shelf_region, (x, 0), (x, shelf_region.shape[0]), (0, 0, 255), 2)

        # Count gaps and put numbers
        for j in range(len(grouped_xs) - 1):
            x_start = grouped_xs[j]
            x_end = grouped_xs[j + 1]
            gap = x_end - x_start
            x_mid = (x_start + x_end) // 2
            y_text = shelf_region.shape[0] // 2
            cv2.putText(
                shelf_region,
                str(gap),
                (x_mid - 10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        shelf_rgb = cv2.cvtColor(shelf_region, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 3))
        plt.imshow(shelf_rgb)
        plt.title(f"Shelf {i+1} - Gap distances between books")
        plt.axis("off")
        plt.show()

# Example usage
image_path = "./image/4.jpg"  # Replace with your image
manual_lines = [440, 1800, 3400]
detect_shelves_manually(image_path, manual_lines)
