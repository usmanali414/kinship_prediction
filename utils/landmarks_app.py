from mtcnn import MTCNN
import cv2
import numpy as numpy
from math import atan2, degrees
from sklearn.preprocessing import normalize
import numpy as np
import logging
# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    angle_rad = atan2(point3[1]-point2[1], point3[0]-point2[0]) - atan2(point1[1]-point2[1], point1[0]-point2[0])
    angle_deg = degrees(angle_rad)
    return angle_deg



def calculate_similarity(image1_landmarks, image2_landmarks):
    # Calculate distances between landmark points in image 1
    image1_distances = []
    for landmark1_name, landmark1 in image1_landmarks.items():
        for landmark2_name, landmark2 in image1_landmarks.items():
            if landmark1_name != landmark2_name:
                distance = calculate_distance(landmark1, landmark2)
                image1_distances.append(distance)

    # Calculate angles between landmark points in image 1
    angle_left_eye_image1 = calculate_angle(image1_landmarks['nose'], image1_landmarks['left_eye'], image1_landmarks['right_eye'])
    angle_right_eye_image1 = calculate_angle(image1_landmarks['nose'], image1_landmarks['right_eye'], image1_landmarks['left_eye'])
    angle_left_mouth_image1 = calculate_angle(image1_landmarks['nose'], image1_landmarks['mouth_left'], image1_landmarks['mouth_right'])
    angle_right_mouth_image1 = calculate_angle(image1_landmarks['mouth_left'], image1_landmarks['mouth_right'], image1_landmarks['nose'])

    # Calculate distances between landmark points in image 2
    image2_distances = []
    for landmark1_name, landmark1 in image2_landmarks.items():
        for landmark2_name, landmark2 in image2_landmarks.items():
            if landmark1_name != landmark2_name:
                distance = calculate_distance(landmark1, landmark2)
                image2_distances.append(distance)

    # Calculate angles between landmark points in image 2
    angle_left_eye_image2 = calculate_angle(image2_landmarks['nose'], image2_landmarks['left_eye'], image2_landmarks['right_eye'])
    angle_right_eye_image2 = calculate_angle(image2_landmarks['nose'], image2_landmarks['right_eye'], image2_landmarks['left_eye'])
    angle_left_mouth_image2 = calculate_angle(image2_landmarks['nose'], image2_landmarks['mouth_left'], image2_landmarks['mouth_right'])
    angle_right_mouth_image2 = calculate_angle(image2_landmarks['mouth_left'], image2_landmarks['mouth_right'], image2_landmarks['nose'])

    # Normalize feature vectors
    image1_features = normalize(np.array(image1_distances + [angle_left_eye_image1, angle_right_eye_image1, angle_left_mouth_image1, angle_right_mouth_image1]).reshape(1, -1))
    image2_features = normalize(np.array(image2_distances + [angle_left_eye_image2, angle_right_eye_image2, angle_left_mouth_image2, angle_right_mouth_image2]).reshape(1, -1))

    # Calculate similarity using cosine similarity
    similarity_score = np.dot(image1_features, image2_features.T)
    # Calculate the rescaled similarity score
    rescaled_score = (similarity_score[0][0] + 1) / 2

    return rescaled_score


def landmarks_points(image_path):
    try:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        detections = detector.detect_faces(img)
        if not detections:
            logging.error("No faces detected in the image.")
            return None
        return detections[0]["keypoints"]
    except Exception as e:
        logging.error(f"Error in detecting landmarks points. {e}")
        return None

def get_landmarks_score(image_path1,image_path2):   
    image1_landmarks = landmarks_points(image_path1)
    image2_landmarks = landmarks_points(image_path2)
    if image1_landmarks and image2_landmarks:
        return calculate_similarity(image1_landmarks, image2_landmarks)
    else:
        logging.error("Could't calculate the landmarks score.")
        return None
