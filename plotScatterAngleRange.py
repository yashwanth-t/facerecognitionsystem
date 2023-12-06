import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
from face_recognition.api import face_locations, face_encodings
import matplotlib.pyplot as plt
X_train = []
y_train = []
labels = {}  # Dictionary to store label mapping
label_counter = 0

for video_name in os.listdir('./inputFaces'):
    # Skip .DS_Store files
    if video_name == '.DS_Store' or 'model.pkl' == video_name or 'labels.pkl' == video_name:
        continue

    print(f'Processing {video_name}...')
    label = video_name

    # Assign a numeric label
    if label not in labels:
        labels[label] = label_counter
        label_counter += 1

    numeric_label = labels[label]
    
    for frame in os.listdir(os.path.join('./inputFaces', video_name)):
        # Skip .DS_Store files
        if frame == '.DS_Store':
            continue

        frame_path = os.path.join('./inputFaces', video_name, frame)
        frame = cv2.imread(frame_path)

        if frame is not None:
            frame_image = frame[0:160, 0:160]
            face_encs = face_encodings(frame_image, [(0, 160, 160, 0)])
            X_train.extend(face_encs)
            y_train.extend([numeric_label] * len(face_encs))
        else:
            print(f"Failed to load image: {frame_path}")

print('inputFaces data extracted.')

# Train SVM
svm_model = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovo')
svm_model.fit(X_train, y_train)

print('SVM trained.')

with open('inputFaces/model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
with open('inputFaces/labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

print('Model and labels saved.')

# ===================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the model and labels
with open('inputFaces/model.pkl', 'rb') as file:
    svm_model = pickle.load(file)
with open('inputFaces/labels.pkl', 'rb') as file:
    labels = pickle.load(file)

# Load X_train and y_train from your dataset or a file if you saved them previously
# Here I assume you have X_train and y_train ready after the face encoding extraction
# If not, you'll need to reconstruct them from your dataset

# Reduce the data to two dimensions using PCA
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)

# Train a new SVM on the 2D data
svm_model_2d = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovo')
svm_model_2d.fit(X_train_2d, y_train)

# Create a mesh grid for the 2D space
xx, yy = np.meshgrid(np.linspace(np.min(X_train_2d[:,0]), np.max(X_train_2d[:,0]), 500),
                     np.linspace(np.min(X_train_2d[:,1]), np.max(X_train_2d[:,1]), 500))

# Predict on each point in the grid
Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Different markers for each class
markers = ['s', 'o', '^', 'x', '*', '+', 'v', '<', '>', 'd']

inverse_labels = {v: k for k, v in labels.items()}

# ... (previous code for PCA and SVM model training)

# Plot the training points with labels
for i, label in enumerate(np.unique(y_train)):
    plt.scatter(X_train_2d[y_train == label, 0], X_train_2d[y_train == label, 1], 
                c=[plt.cm.coolwarm(i / len(np.unique(y_train)))],
                label=inverse_labels[label],  # Use the reversed dictionary here
                marker=markers[i % len(markers)], 
                edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary with PCA-reduced data')
plt.legend()
plt.show()



