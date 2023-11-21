import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
import sys
from face_recognition import face_locations, face_encodings

# Function to extract frames from a video
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    skip_frames = 20
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    print(f'Extracted {len(frames)} frames from {video_path}.')
    return frames

# Extract training data

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    X_train = []
    y_train = []
    for video_name in os.listdir('./training'):
        print(f'Processing {video_name}...')
        label = video_name.split('.')[0]
        frames = extract_frames(f'./training/{video_name}')
        for frame in frames:
            face_locs = face_locations(frame)
            face_encs = face_encodings(frame, face_locs)
            X_train.extend(face_encs)
            y_train.extend([label] * len(face_encs))

    print('Training data extracted.')

    # Train SVM
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    print('SVM trained.')

    # Save the model
    with open('model.pkl', 'wb') as file:
        pickle.dump(svm_model, file)

    print('Model saved.')

with open('model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

print('Model loaded.')

# Function to predict and draw on frames
def predict_and_draw(frame, model):
    face_locs = face_locations(frame)
    face_encs = face_encodings(frame, face_locs)
    for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
        pred = model.predict([face_enc])[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, pred, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    return frame


# Process the test video
test_video_path = './testing/yash_n_batch_2nd_floor.mp4'
cap = cv2.VideoCapture(test_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width,height))

print(f'Processing test video: {test_video_path}...')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = predict_and_draw(frame, svm_model)
    out.write(frame)

print('Test video processed.')

cap.release()
out.release()
