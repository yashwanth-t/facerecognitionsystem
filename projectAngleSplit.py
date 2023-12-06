from math import sqrt
import numpy as np
from cv2 import Laplacian, cvtColor, COLOR_BGR2GRAY, CV_64F, imshow, waitKey, cvtColor, COLOR_BGR2LAB, split
import cv2
from mtcnn.mtcnn import MTCNN
from face_recognition import face_locations, face_encodings
import sys, os
from sklearn.svm import SVC
import pickle

def find_clarity(image):
    return Laplacian(cvtColor(image, COLOR_BGR2GRAY), CV_64F).var()

# Calculate area of triangle by 3 co-ordinates as input
def T_area(A,B,C):
    xa,ya = A
    xb,yb = B
    xc,yc = C
    return float(abs(xa*(yb-yc) + xb*(yc-ya) + xc*(ya-yb)))/2.
# [0,90], [25, 67.5], [50, 45], [75, 22.5], [100, 0]]
def load_left_right_areas(landmarks):
    left_area = T_area(landmarks['left_eye'], landmarks['nose'], landmarks['mouth_left'])
    right_area = T_area(landmarks['right_eye'], landmarks['nose'], landmarks['mouth_right'])
    return left_area, right_area


def find_rotation(landmarks):
    area_direction = 1  # left turned face
    left_area, right_area = load_left_right_areas(landmarks)
    
    if left_area < right_area:
        area_res = left_area / right_area
    else:
        area_direction = -1  # right
        area_res = right_area / left_area
    b0, b1 = 90, -0.9
    return (b0 + b1 * area_res * 100) * area_direction

def detectionNRecognition(frame, frame_number, numeric_label):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    cropped_faces, face_locs, face_angles = [], [], []
    frame1 = frame.copy()

    for face in faces:
        face_bounding_box = face['box']
        left, top, width, height = face_bounding_box
        right = left + width
        bottom = top + height
        face_locs.append((top, right, bottom, left))

        crop_face = frame1[top:bottom, left:right]
        crop_face = cv2.resize(crop_face, (160, 160))
        cropped_faces.append(crop_face)

        face_angle = int(find_rotation(face['keypoints']))
        face_angles.append(face_angle)

        clarity = find_clarity(crop_face)

        # if len(sys.argv) > 1 and sys.argv[1] == 'train':
        #     if clarity < 30:
        #         continue

        if svmDictPredict[-60]:

            for k in svmDict.keys():
                if face_angle <= k:
                    model = svmDictPredict[k]
                    break
            face_encs = face_encodings(frame1, [(top, right, bottom, left)])
            probabilities = model.predict_proba([face_encs[0]])[0]
            label_index = np.argmax(probabilities)
            label = numeric_label[label_index]
            confidence = probabilities[label_index]

            os.makedirs("outputFaces/"+str(label), exist_ok=True)
            face_name = "outputFaces/"+str(label)+'/'+str(face_angle)+".png"
            cv2.imwrite(face_name, crop_face)

            # if clarity is less than 40, skip the frame
            # if clarity < 30:
            #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            #     cv2.putText(frame, f'{label} {confidence:.2f} {clarity:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            #     continue

            cv2.putText(frame, f'Frame: {frame_number}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            if confidence < 0.4:
                # label = 'unknown'
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)



    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        for face_angle, face_loc in zip(face_angles, face_locs):
            face_encs = face_encodings(frame, [face_loc])
            for k in svmDict.keys():
                if face_angle <= k:
                    svmTrainDict[k][0].extend(face_encs)
                    svmTrainDict[k][1].extend([numeric_label] * len(face_encs))
                    break
    return frame

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    skip_frames = 5
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % skip_frames == 0:
            frames.append(frame)
        frame_number += 1
    cap.release()
    print(f'Extracted {len(frames)} frames from {video_path}.')
    return frames


svmDict = {
    -60:'-90to-60.pkl',
    -30:'-60to-30.pkl',
    0:'-30to0.pkl',
    30:'0to30.pkl',
    60:'30to60.pkl',
    90:'60to90.pkl'
}
svmDictPredict = {
    -60:None,
    -30:None,
    0:None,
    30:None,
    60:None,
    90:None
}

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    frame_number = 0
    svmTrainDict = {
        -60:[[], []],
        -30:[[], []],
        0:[[], []],
        30:[[], []],
        60:[[], []],
        90:[[], []]
    }
    X_train = []
    y_train = []
    labels = {}  # Dictionary to store label mapping
    label_counter = 0
    os.remove("./inputFaces/.DS_Store") if os.path.exists("./inputFaces/.DS_Store") else None

    for video_name in os.listdir('./inputFaces'):
        print(f'Processing {video_name}...')
        label = video_name.split('.')[0]

        # Assign a numeric label
        if label not in labels:
            labels[label] = label_counter
            label_counter += 1

        numeric_label = labels[label]
        frames = extract_frames(f'./inputFaces/{video_name}')
        for frame in frames:
            frame = detectionNRecognition(frame, frame_number, numeric_label)
            frame_number += 1
        
    for k, v in svmTrainDict.items():
        svm_model = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovo')
        svm_model.fit(v[0], v[1])
        with open(svmDict[k], 'wb') as file:
            pickle.dump(svm_model, file)

    labels_inv = {v: k for k, v in labels.items()}

    with open('labels.pkl', 'wb') as file:
        pickle.dump(labels_inv, file)

    print('Model and labels saved.')



# Load the model and labels
for k, v in svmDict.items():
    with open(v, 'rb') as file:
        svmDictPredict[k] = pickle.load(file)

with open('labels.pkl', 'rb') as file:
    numeric_label = pickle.load(file)

print('Model and labels loaded.')

# process all test videos in testing folder and save the output video of each video in the output folder
os.remove("./testing/.DS_Store") if os.path.exists("./testing/.DS_Store") else None
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
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % skip_frames == 0:
            frame = detectionNRecognition(frame, frame_number, numeric_label)
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f'Output video saved for {video_name}.')

