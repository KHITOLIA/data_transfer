import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import torch

# --- Configuration ---
EMBEDDING_FILE = 'employee_embeddings_single.npz'
# **CRITICAL:** Threshold for Cosine Similarity. 
# Lower value = Higher Security. Tune this carefully.
COSINE_THRESHOLD = 0.3 

# Placeholder for a live face image (Replace with actual webcam capture later)
# Use an image of an employee (e.g., Alice) or a non-employee (e.g., Stranger.jpg) to test the logic
TEST_IMAGE_PATH = 'employee_images/tushar_1.jpg' 
# or 
# TEST_IMAGE_PATH = 'single_dataset/Stranger.jpg' 

# --- Initialize Models ---
print("Initializing models and loading database...")
# MTCNN and FaceNet initialisation remains the same
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True
).eval()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# --- Load Biometric Template Database ---
try:
    data = np.load(EMBEDDING_FILE, allow_pickle=True)
    DATABASE_EMBEDDINGS = data['arr_0']
    DATABASE_LABELS = data['arr_1']
    print(f"Loaded {len(DATABASE_LABELS)} employee templates.")
except FileNotFoundError:
    print(f"ERROR: Embedding file not found at '{EMBEDDING_FILE}'.")
    exit()

# --- Live Face Processing Function ---
def get_live_face_embedding(img_path):
    """Detects face and generates a single 512-dim embedding from the live image."""
    try:
        img = Image.open(img_path).convert('RGB')
        face_tensor = mtcnn(img)

        if face_tensor is None:
            return None

        with torch.no_grad():
            live_embedding = resnet(face_tensor.unsqueeze(0)).numpy().flatten()
            
        return live_embedding

    except Exception as e:
        print(f"   [ERROR] Processing live image: {e}")
        return None

# --- Step 2: Perform 1:N Identification (Secure Authentication) ---

# 2a. Generate embedding for the live image
print(f"\nProcessing live face from: {TEST_IMAGE_PATH}...")
live_embedding = get_live_face_embedding(TEST_IMAGE_PATH)

if live_embedding is None:
    # Handle case where NO face is detected
    print("\n🚨 Authentication Failed: No valid face detected in the live stream.")
else:
    # 2b. Calculate similarity (distance) to all database templates
    distances = []
    for db_embedding in DATABASE_EMBEDDINGS:
        # Cosine distance: 0 = perfect match, 2 = perfect mismatch
        dist = cosine(live_embedding, db_embedding)
        distances.append(dist)
    
    # Find the closest match index
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    predicted_employee_id = DATABASE_LABELS[min_distance_index]
    
    print("-" * 50)
    print(f"Closest match found in database: {predicted_employee_id}")
    print(f"Minimum Cosine Distance (Proximity): {min_distance:.4f}")
    print(f"Security Threshold: {COSINE_THRESHOLD}")
    print("-" * 50)
    
    # 2c. Apply the threshold for final decision (Secure Check)
    if min_distance <= COSINE_THRESHOLD:
        # **The Match is Secure:** Grant access and identify the person
        print(f"\n✅ AUTHENTICATION SUCCESS: Welcome, {predicted_employee_id}!")
    else:
        # **The Match is NOT Secure:** Deny access and do NOT identify the person
        print("\n❌ AUTHENTICATION FAILED: **Not Matched.**")

print("-" * 50)