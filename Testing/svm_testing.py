import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


mediapipe_csv_path = '../Dataset/MediaPipe_dataset/mediapipe_data.csv'
hog_csv_path = '../Dataset/MediaPipe_dataset/mediapipe_hog_data.csv'

distance_data = read_csv(mediapipe_csv_path)
hog_data = read_csv(hog_csv_path)

# Merge data so that both models use same image data 
merged_data = pd.merge(distance_data, hog_data, on='id', suffixes=('_landmark', '_hog'))
# Use only one label column
merged_data = merged_data.rename(columns={'label_landmark': 'label'}).drop('label_hog', axis=1)

X_distance = merged_data['landmark_distances'].apply(eval).tolist()  
X_hog = merged_data['hog_descriptor'].apply(eval).tolist()            
Y = merged_data['label'].tolist()

# Create same test and train data split
indices = np.arange(len(Y))
idx_train, idx_test, y_train, y_test = train_test_split(indices, Y, test_size=0.2, stratify=Y, random_state=42)

# Train and test data for distance implementation
X_train_distances = [X_distance[i] for i in idx_train]
X_test_distances  = [X_distance[i] for i in idx_test]

# Train and test data for hog implementation
X_train_hog = [X_hog[i] for i in idx_train]
X_test_hog  = [X_hog[i] for i in idx_test]

# Test results for distance model
print("Distance model:")
svm_distances = SVC(kernel='linear', C=1.0, random_state=42)
svm_distances.fit(X_train_distances, y_train)
pred_distance = svm_distances.predict(X_test_distances)
accuracy_distance = accuracy_score(y_test, pred_distance)
precision_distance = precision_score(y_test, pred_distance, average='macro')
recall_distance = recall_score(y_test, pred_distance, average='macro')

print(f'Accuracy: {accuracy_distance * 100:.2f}%')
print(f'Precision (macro average): {precision_distance * 100:.2f}%')
print(f'Recall (macro average): {recall_distance * 100:.2f}%')

print()

# Using PCA
# sc = StandardScaler()
# X_train_hog = sc.fit_transform(X_train_hog)
# X_test_hog = sc.transform(X_test_hog)
# pca = PCA(n_components=50)
# X_train_hog = pca.fit_transform(X_train_hog)
# X_test_hog = pca.transform(X_test_hog)

# Test results for hog model
print("HOG model:")
svm_hog = SVC(kernel='linear', C=1.0, random_state=42)
svm_hog.fit(X_train_hog, y_train)
pred_hog = svm_hog.predict(X_test_hog)
accuracy_hog = accuracy_score(y_test, pred_hog)
accuracy_hog = accuracy_score(y_test, pred_hog)
precision_hog = precision_score(y_test, pred_hog, average='macro')
recall_hog = recall_score(y_test, pred_hog, average='macro')

print(f'Accuracy: {accuracy_hog * 100:.2f}%')
print(f'Precision (macro average): {precision_hog * 100:.2f}%')
print(f'Recall (macro average): {recall_hog * 100:.2f}%')

