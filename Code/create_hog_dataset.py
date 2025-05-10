import os
import csv
import cv2
import numpy as np
import scipy.signal
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='./hand_landmarker.task')
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1)
detector = vision.HandLandmarker.create_from_options(hand_landmarker_options)

raw_dataset_path = '../Dataset/RAW_dataset/'

all_hog_descriptors = []
all_labels = []

# Used for testing
all_ids = []
id = 0

for path, folders, files in os.walk(raw_dataset_path):
    for folder in folders:
        folder_path = os.path.join(path, folder)
        print(f'Processing folder: {folder}')
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                image_path = os.path.join(folder_path, file)
                image = mp.Image.create_from_file(image_path)
                hand_landmarks = detector.detect(image).hand_landmarks
                if hand_landmarks:
                    # Used for testing
                    id += 1

                    # Get hand bounding box
                    x_vals = []
                    y_vals = []
                    for hand_landmark in hand_landmarks:
                        for landmark in hand_landmark:
                            # Extract x, y coordinates of each landmark and append to corresponding list
                            x_vals.append(landmark.x)
                            y_vals.append(landmark.y)

                    # Get min and max values for bounding box
                    x_min = min(x_vals)
                    y_min = min(y_vals)
                    x_max = max(x_vals)
                    y_max = max(y_vals)
                    
                    # Convert to pixel coordinates using 
                    height, width, _ = image.numpy_view().shape
                    x_min_pixel = int(x_min * width)
                    x_max_pixel = int(x_max * width)
                    y_min_pixel = int(y_min * height)
                    y_max_pixel = int(y_max * height)
                    
                    # Crop hand region using bounding box
                    hand_img = image.numpy_view()[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                    # In case of invalid hand image...
                    if hand_img.size == 0:
                        continue
                    
                    # Resize hand image for consistency and convert to grayscale
                    hand_img = cv2.resize(hand_img, (256, 256))
                    hand_img = hand_img[:, :, 0]
                    hand_img = hand_img.astype(np.float32)
                    
                    # Compute gradients
                    ix = np.matmul(np.array([[1], [2], [1]]), np.array([[1, 0, -1]]))
                    iy = np.matmul(np.array([[1], [0], [-1]]), np.array([[1, 2, 1]]))
                    deriv_x = scipy.signal.convolve(hand_img, ix, mode='same')
                    deriv_y = scipy.signal.convolve(hand_img, iy, mode='same')
                    grad_mag = np.sqrt(deriv_x**2 + deriv_y**2)
                    grad_ang = np.arctan2(deriv_y, deriv_x)

                    # Convert angles to degrees between 0 to 180
                    grad_ang_deg = np.degrees(grad_ang) % 180

                    # HoG implementation

                    # HoG variables
                    cell_size = 8
                    num_bins = 9
                    bin_size = 20
                    all_histograms = []
                    num_rows = hand_img.shape[0] // cell_size
                    num_cols = hand_img.shape[1] // cell_size

                    # Iterate through all 8x8 cells
                    for i in range(num_rows):
                        for j in range(num_cols):
                            histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                            # Iterate through single cell
                            for r in range(i*cell_size, i*cell_size+cell_size):
                                for c in range(j*cell_size, j*cell_size+cell_size):
                                    mag = grad_mag[r, c]
                                    ang = grad_ang_deg[r, c]

                                    # Get closest bin index and bin
                                    bin_i = round(ang / bin_size)
                                    if bin_i == 9:
                                        bin_i = 0
                                        ang -= 180
                                    bin_at_i = bin_i * bin_size

                                    # Use interpolation
                                    percent_in_neighbor_bin = abs((ang - bin_at_i) / bin_size)
                                    percent_in_bin = 1-percent_in_neighbor_bin

                                    histogram[bin_i] += mag*percent_in_bin
                                    if ang - bin_at_i < 0:
                                        if bin_i - 1 >= 0:
                                            histogram[bin_i-1] += mag*percent_in_neighbor_bin
                                        else: # Loop around bins
                                            histogram[num_bins-1] += mag*percent_in_neighbor_bin
                                    elif ang - bin_at_i > 0:
                                        if bin_i + 1 < num_bins:
                                            histogram[bin_i+1] += mag*percent_in_neighbor_bin
                                        else: # Loop around bins
                                            histogram[0] += mag*percent_in_neighbor_bin

                            all_histograms.append(histogram)

                    # Append data for this image to all_hog_descriptors and all_labels lists
                    all_hog_descriptors.append([i for s in all_histograms for i in s])
                    all_labels.append(folder)

                    # Used for testing
                    all_ids.append(id)


# Save to a CSV file
csv_file_path = '../Dataset/MediaPipe_dataset/mediapipe_hog_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'hog_descriptor', 'label'])
    for id, hog_descriptor, label in zip(all_ids, all_hog_descriptors, all_labels):     
        writer.writerow([id, hog_descriptor, label])
print(f'CSV file {csv_file_path} created successfully.')

# Note: do not open .csv file with Excel, too many characters are in the hog_descriptors