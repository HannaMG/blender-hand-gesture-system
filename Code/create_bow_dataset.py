import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# Setup MediaPipe HandLandmarker.
base_options = python.BaseOptions(model_asset_path='./hand_landmarker.task')
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1)
detector = vision.HandLandmarker.create_from_options(hand_landmarker_options)

# Set raw dataset path.
raw_dataset_path = '../Dataset/RAW_dataset/'

# -----------------------
# First pass: Build a vocabulary.
# We will use ORB to extract local descriptors from the hand region.
orb = cv2.ORB_create()
all_descriptors_list = []  # will collect descriptors from all images

# Loop over images to collect descriptors.
for path, folders, files in os.walk(raw_dataset_path):
    for folder in folders:
        folder_path = os.path.join(path, folder)
        print(f'Collecting descriptors from folder: {folder}')
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                image_path = os.path.join(folder_path, file)
                image = mp.Image.create_from_file(image_path)
                # Run hand detection.
                hand_landmarks = detector.detect(image).hand_landmarks
                if hand_landmarks:
                    # Compute bounding box from landmarks.
                    landmarks_data = []
                    for hand_landmark in hand_landmarks:
                        for landmark in hand_landmark:
                            landmarks_data.append([landmark.x, landmark.y, landmark.z])
                    x_vals = [l[0] for l in landmarks_data]
                    y_vals = [l[1] for l in landmarks_data]
                    x_min = min(x_vals)
                    y_min = min(y_vals)
                    x_max = max(x_vals)
                    y_max = max(y_vals)
                    # Convert normalized coords to pixels.
                    np_image = image.numpy_view()
                    h_img, w_img, _ = np_image.shape
                    x_min_pixel = int(x_min * w_img)
                    x_max_pixel = int(x_max * w_img)
                    y_min_pixel = int(y_min * h_img)
                    y_max_pixel = int(y_max * h_img)
                    # Crop hand region.
                    hand_img = np_image[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                    if hand_img.size == 0:
                        continue
                    # Resize to fixed size 128x128.
                    hand_img = cv2.resize(hand_img, (128, 128))
                    # Convert to grayscale.
                    hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_RGB2GRAY)
                    # Extract ORB descriptors.
                    keypoints, des = orb.detectAndCompute(hand_gray, None)
                    if des is not None:
                        all_descriptors_list.append(des)

# If no descriptors were found, exit.
if len(all_descriptors_list) == 0:
    print("No descriptors found in the dataset!")
    exit()

# Stack all descriptors into one array.
all_descriptors = np.vstack(all_descriptors_list)
print("Total descriptors collected:", all_descriptors.shape[0])

# Cluster the descriptors via k-means to form the vocabulary.
# (Set the desired number of visual words, e.g., 50)
K = 50
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels_k, vocabulary = cv2.kmeans(all_descriptors.astype(np.float32), 
                                               K, None, criteria, 10, flags)
print("Vocabulary shape:", vocabulary.shape)
# vocabulary now is a (K x descriptor_dim) array.

# -----------------------
# Second pass: Compute BoW feature vectors (histograms) for each image.
all_bow_features = []
all_labels = []

for path, folders, files in os.walk(raw_dataset_path):
    for folder in folders:
        folder_path = os.path.join(path, folder)
        print(f'Computing BoW for folder: {folder}')
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                image_path = os.path.join(folder_path, file)
                image = mp.Image.create_from_file(image_path)
                hand_landmarks = detector.detect(image).hand_landmarks
                if hand_landmarks:
                    # Compute bounding box from landmarks.
                    landmarks_data = []
                    for hand_landmark in hand_landmarks:
                        for landmark in hand_landmark:
                            landmarks_data.append([landmark.x, landmark.y, landmark.z])
                    x_vals = [l[0] for l in landmarks_data]
                    y_vals = [l[1] for l in landmarks_data]
                    x_min = min(x_vals)
                    y_min = min(y_vals)
                    x_max = max(x_vals)
                    y_max = max(y_vals)
                    np_image = image.numpy_view()
                    h_img, w_img, _ = np_image.shape
                    x_min_pixel = int(x_min * w_img)
                    x_max_pixel = int(x_max * w_img)
                    y_min_pixel = int(y_min * h_img)
                    y_max_pixel = int(y_max * h_img)
                    hand_img = np_image[y_min_pixel:y_max_pixel, x_min_pixel:x_max_pixel]
                    if hand_img.size == 0:
                        continue
                    hand_img = cv2.resize(hand_img, (128, 128))
                    hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_RGB2GRAY)
                    
                    # Extract ORB descriptors for this image.
                    kp, des = orb.detectAndCompute(hand_gray, None)
                    # Create a histogram for this image.
                    bow_hist = np.zeros(K)
                    if des is not None:
                        # For each ORB descriptor, find its nearest cluster center (visual word).
                        for d in des:
                            # Compute Euclidean distances between the descriptor and each vocabulary word.
                            distances = np.linalg.norm(vocabulary - d, axis=1)
                            word_idx = np.argmin(distances)
                            bow_hist[word_idx] += 1
                    # Normalize the histogram (L1 normalization).
                    if np.sum(bow_hist) != 0:
                        bow_hist = bow_hist / np.sum(bow_hist)
                    all_bow_features.append(bow_hist)
                    all_labels.append(folder)

# -----------------------
# Save the BoW features to a CSV file.
# Here, the entire BoW feature vector is saved as one string under "bow_descriptor", and the label goes into "label"
csv_file_path = '../Dataset/MediaPipe_dataset/mediapipe_bow_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['bow_descriptor', 'label'])
    for feat, label in zip(all_bow_features, all_labels):
        writer.writerow([feat.tolist(), label])
print(f'CSV file {csv_file_path} created successfully.')