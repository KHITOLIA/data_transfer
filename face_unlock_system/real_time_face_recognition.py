import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import os
from image_embeddings import face_embeddings

# --- Configuration ---
EMBEDDING_FILE = 'employee_embeddings_single.npz'
COSINE_THRESHOLD = 0.95
FRAME_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_SUCCESS = (0, 255, 0) # Green
COLOR_FAIL = (0, 0, 255)    # Red

# --- Load Database and Models (One Time) ---
try:
    data = np.load(EMBEDDING_FILE, allow_pickle=True)
    DATABASE_EMBEDDINGS = data['arr_0']
    DATABASE_LABELS = data['arr_1']
    print(f"Loaded {len(DATABASE_LABELS)} employee templates.")
except FileNotFoundError:
    print(f"FATAL ERROR: Embedding file not found at '{EMBEDDING_FILE}'. Run the enrollment script first.")
    exit()

# Initialize Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Initializing MTCNN and FaceNet models on {device}...")

detector = MTCNN(
    margin=20, min_face_size=20, factor=0.709,
    device=device
).eval()

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# --- Core Function: Get Embedding from Live Frame ---
def get_embedding_from_frame(face_tensor):
    """Generates embedding from the already pre-processed face tensor."""
    try:
        if face_tensor is None:
            return None
            
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0)).cpu().numpy().flatten()
            
        return embedding
    except Exception as e:
        # print(f"Embedding error: {e}") 
        return None

# --- Webcam Loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting continuous real-time authentication. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # 1. Face Detection (MTCNN): Get Boxes and Probabilities
    boxes_list, probs_list = detector.detect(pil_img) 
    
    # Check if any faces were detected
    if boxes_list is not None and len(boxes_list) > 0:
        
        # 2. Alignment and Embedding Generation for ALL detected faces
        aligned_face_tensors = detector(pil_img) 

        for i, box in enumerate(boxes_list):
            x, y, x2, y2 = [int(i) for i in box]
            
            # 3. Generate Live Embedding
            live_embedding = get_embedding_from_frame(aligned_face_tensors[i])
            
            name = "Not an authorized person"
            display_color = COLOR_FAIL

            if live_embedding is not None:
                distances = []
                for db_embedding in DATABASE_EMBEDDINGS:
                    dist = cosine(live_embedding, db_embedding)
                    distances.append(dist)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                predicted_employee_name = DATABASE_LABELS[min_distance_index]

                # --- TEMPORARY DEBUG PRINT: Check the actual distance ---
                print(f"DEBUG: Closest: {predicted_employee_name}, Min Distance: {min_distance:.4f}")
                # --- REMOVE THIS LINE AFTER YOU DETERMINE YOUR THRESHOLD ---

                # 5. Apply Security Threshold
                if min_distance <= COSINE_THRESHOLD:
                    # SUCCESS
                    name = predicted_employee_name
                    display_color = COLOR_SUCCESS
                # Else: name remains "Not an authorized person"

            # 6. Display Result on Frame
            cv2.rectangle(frame, (x, y), (x2, y2), display_color, FRAME_THICKNESS)
            
            # Set up text background
            text_size = cv2.getTextSize(name , FONT, 0.7, FRAME_THICKNESS)[0]
            text_x = x
            text_y = y - 10
            
            cv2.rectangle(frame, (text_x, text_y - text_size[1]), 
                          (text_x + text_size[0], text_y + 5), 
                          display_color, cv2.FILLED)
            
            cv2.putText(frame, name, (text_x, text_y), FONT, 0.7, (0, 0, 0), FRAME_THICKNESS, cv2.LINE_AA)

    # Display the final frame
    cv2.imshow('Biometric Authentication System', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()