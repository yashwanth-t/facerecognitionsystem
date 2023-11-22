import cv2
import os
import numpy as np
import pickle
from sklearn.svm import SVC
import sys
from face_recognition import face_locations, face_encodings
from cv2 import Laplacian, cvtColor, COLOR_BGR2GRAY, CV_64F, imshow, waitKey, cvtColor, COLOR_BGR2LAB, split

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    skip_frames = 5
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

def find_clarity(image):
    return Laplacian(cvtColor(image, COLOR_BGR2GRAY), CV_64F).var()

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    X_train = []
    y_train = []
    labels = {}  # Dictionary to store label mapping
    label_counter = 0

    for video_name in os.listdir('./training'):
        print(f'Processing {video_name}...')
        label = video_name.split('.')[0]

        # Assign a numeric label
        if label not in labels:
            labels[label] = label_counter
            label_counter += 1

        numeric_label = labels[label]
        frames = extract_frames(f'./training/{video_name}')
        for frame in frames:
            face_locs = face_locations(frame)

            skipped_frames = 0

            # crop the face using the face locations
            for (top, right, bottom, left) in face_locs:
                frame_image = frame[top:bottom, left:right]
                # find the clarity of the image
                clarity = find_clarity(frame_image)
                # if clarity is less than 40, skip the frame
                if clarity < 40:

                    continue

            face_encs = face_encodings(frame, face_locs)
            X_train.extend(face_encs)
            y_train.extend([numeric_label] * len(face_encs))

    print('Training data extracted.')

    # Train SVM
    svm_model = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovo')
    svm_model.fit(X_train, y_train)

    print('SVM trained.')

    # Create a reverse mapping for labels
    labels_inv = {v: k for k, v in labels.items()}

    # Save the model and labels
    with open('model.pkl', 'wb') as file:
        pickle.dump(svm_model, file)
    with open('labels.pkl', 'wb') as file:
        pickle.dump(labels_inv, file)

    print('Model and labels saved.')

# Load the model and labels
with open('model.pkl', 'rb') as file:
    svm_model = pickle.load(file)
with open('labels.pkl', 'rb') as file:
    labels_inv = pickle.load(file)

print('Model and labels loaded.')

def predict_and_draw(frame, model, labels, frame_count):
    face_locs = face_locations(frame)
    face_encs = face_encodings(frame, face_locs)
    for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):

        # add frame number to the frame
        cv2.putText(frame, f'Frame: {frame_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # crop the face using the face locations
        face_frame = frame[top:bottom, left:right]
        # find the clarity of the image
        clarity = find_clarity(face_frame)
        
        probabilities = model.predict_proba([face_enc])[0]
        label_index = np.argmax(probabilities)
        label = labels[label_index]
        confidence = probabilities[label_index]

        # if clarity is less than 40, skip the frame
        if clarity < 40:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f} {clarity:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            continue

        if confidence < 0.4:
            # label = 'unknown'
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return frame


# process all test videos in testing folder and save the output video of each video in the output folder
for video_name in os.listdir('./testing'):
    print(f'Processing {video_name}...')
    test_video_path = f'./testing/{video_name}'
    cap = cv2.VideoCapture(test_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'./output/{video_name}', fourcc, fps, (width, height))

    skip_frames = 1
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frame = predict_and_draw(frame, svm_model, labels_inv, frame_count=frame_count)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f'Output video saved for {video_name}.')
