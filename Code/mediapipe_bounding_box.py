import cv2
import mediapipe as mp
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def visualize_hog(hog_cells, cell_size=8, scale=40):
    n_rows, n_cols, num_bins = hog_cells.shape
    hog_vis_img = np.zeros((n_rows * cell_size, n_cols * cell_size), dtype=np.uint8)
    bin_size = 180 / num_bins  # Each bin covers 20° for 9 bins

    # Loop over each cell.
    for r in range(n_rows):
        for c in range(n_cols):
            cell_hist = hog_cells[r, c, :]
            # Determine the cell center in the overall visualization image.
            center_x = int(c * cell_size + cell_size / 2)
            center_y = int(r * cell_size + cell_size / 2)
            # For each bin, draw a line representing the histogram vote.
            for bin_idx in range(num_bins):
                magnitude = cell_hist[bin_idx]
                # Compute the bin’s center orientation in degrees.
                angle_deg = bin_idx * bin_size + bin_size / 2.0
                angle_rad = np.deg2rad(angle_deg)
                
                # Use the magnitude * scale as the line length.
                line_length = magnitude * scale
                # Calculate endpoints (centered at the cell center).
                x1 = int(center_x - (line_length / 2) * np.cos(angle_rad))
                y1 = int(center_y - (line_length / 2) * np.sin(angle_rad))
                x2 = int(center_x + (line_length / 2) * np.cos(angle_rad))
                y2 = int(center_y + (line_length / 2) * np.sin(angle_rad))
                cv2.line(hog_vis_img, (x1, y1), (x2, y2), color=255, thickness=1)
    return hog_vis_img

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect landmark coordinates.
                x_vals = [landmark.x for landmark in hand_landmarks.landmark]
                y_vals = [landmark.y for landmark in hand_landmarks.landmark]
                x_min, y_min = min(x_vals), min(y_vals)
                x_max, y_max = max(x_vals), max(y_vals)

                height, width, _ = image.shape
                x_min_pixel = int(x_min * width)
                x_max_pixel = int(x_max * width)
                y_min_pixel = int(y_min * height)
                y_max_pixel = int(y_max * height)

                cv2.rectangle(image, (x_min_pixel, y_min_pixel), (x_max_pixel, y_max_pixel), (0, 255, 0), 2)

                hand_image = image[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                if hand_image is not None and hand_image.size != 0:
                    # Resize image for consistency.
                    hand_image = cv2.resize(hand_image, (256, 256))
                    # Convert to grayscale (first channel).
                    hand_image = hand_image[:, :, 0].astype(np.float32)

                    # Compute gradients.
                    ix = np.matmul(np.array([[1], [2], [1]]), np.array([[1, 0, -1]]))
                    iy = np.matmul(np.array([[1], [0], [-1]]), np.array([[1, 2, 1]]))
                    deriv_x = scipy.signal.convolve(hand_image, ix, mode='same')
                    deriv_y = scipy.signal.convolve(hand_image, iy, mode='same')
                    grad_mag = np.sqrt(deriv_x**2 + deriv_y**2)
                    grad_ang = np.arctan2(deriv_y, deriv_x)

                    # Convert angles to degrees in [0, 180).
                    grad_ang_deg = np.degrees(grad_ang)
                    grad_ang_deg[grad_ang_deg < 0] += 180

                    # Divide image into 8x8 cells.
                    num_bins = 9
                    cell_size = 8
                    bin_size = 20  # each bin spans 20 degrees.
                    n_rows = hand_image.shape[0] // cell_size  # should be 16 for 128x128
                    n_cols = hand_image.shape[1] // cell_size
                    all_histograms = np.zeros((n_rows, n_cols, num_bins))

                    # Compute histogram for each cell using interpolation.
                    for i in range(n_rows):
                        for j in range(n_cols):
                            cell_mag = grad_mag[i*cell_size:(i+1)*cell_size,
                                                j*cell_size:(j+1)*cell_size]
                            cell_ang = grad_ang_deg[i*cell_size:(i+1)*cell_size,
                                                    j*cell_size:(j+1)*cell_size]
                            histogram = np.zeros(num_bins)

                            for r in range(cell_mag.shape[0]):
                                for c in range(cell_mag.shape[1]):
                                    magnitude = cell_mag[r, c]
                                    angle = cell_ang[r, c]
                                    bin_idx = int(angle // bin_size)
                                    if bin_idx >= num_bins:
                                        bin_idx = num_bins - 1
                                    bin_center = bin_idx * bin_size + (bin_size / 2.0)
                                    diff = (angle - bin_center) / bin_size
                                    weight_primary = 1 - abs(diff)
                                    histogram[bin_idx] += magnitude * weight_primary
                                    if diff > 0 and bin_idx < num_bins - 1:
                                        histogram[bin_idx + 1] += magnitude * abs(diff)
                                    elif diff < 0 and bin_idx > 0:
                                        histogram[bin_idx - 1] += magnitude * abs(diff)
                            all_histograms[i, j, :] = histogram

                    print("Max histogram vote:", np.max(all_histograms))
                    # Normalize histogram values for visualization.
                    all_histograms_norm = all_histograms / np.max(all_histograms)
                    print("Max normalized vote:", np.max(all_histograms_norm))
                    
                    # Visualize HOG.
                    hog_vis = visualize_hog(all_histograms_norm, cell_size=cell_size, scale=30)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(hog_vis, cmap='gray')
                    plt.title("HOG Visualization")
                    plt.axis('off')
                    plt.show()

        cv2.imshow("Hand Bounding Box", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()