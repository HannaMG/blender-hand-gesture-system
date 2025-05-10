import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from os import listdir
import csv

base_options = python.BaseOptions(model_asset_path='./hand_landmarker.task')
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1)
detector = vision.HandLandmarker.create_from_options(hand_landmarker_options)

raw_dataset_path = '../Dataset/RAW_dataset/'

all_distance_vectors = []
all_labels = []

# Used for testing
all_ids = []
id = 0

for path, folders, files in os.walk(raw_dataset_path):
    for folder in folders:
        folder_path = os.path.join(path, folder)
        print(f'Processing folder: {folder}')
        for file in listdir(folder_path):
            if file.endswith('.jpg'):
                image_path = os.path.join(folder_path, file)
                image = mp.Image.create_from_file(image_path)
                hand_landmarks = detector.detect(image).hand_landmarks
                if hand_landmarks:
                    landmarks_data = []
                    distance_vector = []

                    # Used for testing
                    id += 1

                    for hand_landmark in hand_landmarks:
                        for landmark in hand_landmark:
                            # Extract x, y, z coordinates of each landmark and append to landmarks_data list
                            landmarks_data.append([landmark.x, landmark.y, landmark.z]) 

                    # Calculate distances between landmarks
                    for i in range(len(landmarks_data)):
                        for j in range(i + 1, len(landmarks_data)):
                            distance = ((landmarks_data[i][0] - landmarks_data[j][0]) ** 2 +
                                        (landmarks_data[i][1] - landmarks_data[j][1]) ** 2 +
                                        (landmarks_data[i][2] - landmarks_data[j][2]) ** 2) ** 0.5
                            distance_vector.append(distance)

                    # Get min and max distances
                    min_distance = min(distance_vector)
                    max_distance = max(distance_vector)

                    # Normalize the distances (min-max normalization)
                    for i in range(len(distance_vector)):
                        distance_vector[i] = (distance_vector[i] - min_distance) / (max_distance - min_distance)
                    
                    # Append data for this image to all_landmarks and all_labels lists
                    all_distance_vectors.append(distance_vector)
                    all_labels.append(folder)

                    # Used for testing
                    all_ids.append(id)

# Save to a CSV file
csv_file_path = '../Dataset/MediaPipe_dataset/mediapipe_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'landmark_distances', 'label'])
    writer.writerows(zip(all_ids, all_distance_vectors, all_labels))
print(f'CSV file {csv_file_path} created successfully.')




