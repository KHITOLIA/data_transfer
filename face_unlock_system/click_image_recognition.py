import os
import cv2
import numpy as np
import torch
from PIL import Image
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import time
from image_embeddings import face_embeddings


# Ask for person's name
# person_name = input("Enter the person's name: ").strip()



# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
detector = MTCNN()

count = 0
MAX_IMAGES = 1 # Number of face images to capture

print(f"[INFO] Starting capture for Image please Look at the camera...")



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        # Padding factor (percentage of face size)
        pad_x = int(w * 0.2)   # 20% of width
        pad_y = int(h * 0.2)   # 20% of height
        
        # Apply padding
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        # Crop expanded face
        face_img = frame[y1:y2, x1:x2]
    
        
        # # Crop the face
        face_img = frame[y:y+h+30, x:x+w+10]

        # Save only if we have a valid face crop
        if face_img.size > 0:
            face_resized = cv2.resize(face_img, (160, 160))  # FaceNet size
            count += 1
            file_path = 'employee.jpg'
            cv2.imwrite(file_path, face_resized)
    # Break if enough images captured
    if count >= MAX_IMAGES:
        print("[INFO] Capture complete!")
        break

    # # Press 'q' to quit early
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()


embedding_file = 'employee_embeddings_single.npz'

cosing_threshold = 0.35

# test_img_path = 'employee.jpg'

try:
    data = np.load(embedding_file, allow_pickle=True)
    DATABASE_EMBEDDINGS = data['arr_0']
    DATABASE_LABELS = data['arr_1']
    # print(f"Loaded {len(DATABASE_LABELS)} employee templates.")
except FileNotFoundError:
    print(f"ERROR: Embedding file not found at '{embedding_file}'.")
    exit()

image_embedding = face_embeddings(file_path)

if image_embedding is None:
    print("you failed as a human :) ")

else:
    distances = []

    for db_embedding in DATABASE_EMBEDDINGS:

        dist = cosine(image_embedding, db_embedding)
        distances.append(dist)
    
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    predicted_employee_name = DATABASE_LABELS[min_distance_index]
    print("-"*50)

    # 2c. Apply the threshold for final decision (Secure Check)
    if min_distance <= cosing_threshold:
        # **The Match is Secure:** Grant access and identify the person
        print(f"\n✅ AUTHENTICATION SUCCESS: Welcome, {predicted_employee_name}!")
    else:
        # **The Match is NOT Secure:** Deny access and do NOT identify the person
        print("\n❌ AUTHENTICATION FAILED: **Not Matched.**")

print("-" * 50)

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_path}' deleted successfully.")
else:
    print(f"File '{file_path}' does not exist.")