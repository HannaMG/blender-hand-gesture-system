from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.utils import shuffle
import joblib


csv_file_path = '../Dataset/MediaPipe_dataset/mediapipe_data.csv'
data = read_csv(csv_file_path)

X = data['landmark_distances'].apply(eval).tolist()  
Y = data['label'].tolist()

X, Y = shuffle(X, Y)  # Shuffle the data

# Create and train the SVM model
svm_model = SVC(kernel='linear', C=1.0) 
svm_model.fit(X, Y)

# Test the model
print("Distance model:")
accuracy = svm_model.score(X, Y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(svm_model, 'mediapipe_svm_model.pkl')
print("Model saved as 'mediapipe_svm_model.pkl'")

csv_file_path = '../Dataset/MediaPipe_dataset/mediapipe_hog_data.csv'
data = read_csv(csv_file_path)

X = data['hog_descriptor'].apply(eval).tolist()  
Y = data['label'].tolist()

X, Y = shuffle(X, Y)  # Shuffle the data

# Create and train the SVM model
svm_model = SVC(kernel='linear', C=1.0) 
svm_model.fit(X, Y)

# Test the model
print()
print("HOG model:")
accuracy = svm_model.score(X, Y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(svm_model, 'mediapipe_hog_svm_model.pkl')
print("Model saved as 'mediapipe_hog_svm_model.pkl'")



