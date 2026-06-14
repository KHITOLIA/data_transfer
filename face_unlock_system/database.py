import os
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from facenet_pytorch import MTCNN, InceptionResnetV1
# Assuming 'from image_embeddings import face_embeddings' 
# is replaced by the actual function definition below for completeness.

# --- Configuration ---
data_folder = 'employee_images'
EMBEDDING_FILE = 'employee_embeddings_single.npz' # The file where data is saved

# --- Initialize Models (Once) ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Initializing MTCNN and FaceNet models...")

MTCNN_DETECTOR = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
).eval()

RESNET_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models have been initialized successfully.")
os.makedirs(data_folder, exist_ok=True) # Ensure the image folder exists


# --- Core Function: Generate Face Embedding ---
def face_embeddings(filepath):
    """Detects, aligns, and generates the embedding for a single face."""
    try:
        img = Image.open(filepath).convert('RGB')
        
        # MTCNN detects, aligns, and crops the face to a 160x160 tensor
        aligned_face_tensor = MTCNN_DETECTOR(img)

        if aligned_face_tensor is None:
            return None

        # FaceNet generates the 512-D embedding
        with torch.no_grad():
            embedding = RESNET_MODEL(aligned_face_tensor.unsqueeze(0)).cpu().numpy().flatten()
            
        return embedding

    except Exception:
        return None
# --- End of Core Function ---


# --- Step 1 & 2: Process Images and Extract Embeddings ---
employee_data = defaultdict(list)
valid_extensions = ('.png', '.jpg', '.jpeg')

print(f"\nScanning and processing images in {data_folder}")

for filename in os.listdir(data_folder):
    if filename.lower().endswith(valid_extensions):
        filepath = os.path.join(data_folder, filename)

        try:
            # Assumes format: EmployeeName_... .ext
            employee_name = filename.split("_")[0]
        except Exception:
            print(f"   [SKIP] Invalid filename format (no underscore): {filename}")
            continue

        embedding = face_embeddings(filepath)

        if embedding is not None:
            employee_data[employee_name].append(embedding)
            print(f"   [OK] {employee_name} - {filename}")
        else:
            print(f"   [FAIL] {employee_name} - {filename}: No valid face found.")

# --- Step 3 & 4: Aggregate Templates ---
final_embeddings = []
final_labels = []

for employee_name, embeddings_list in employee_data.items():
    if len(embeddings_list) > 0:
        # Biometric Template: Calculate the MEAN embedding
        mean_embedding = np.mean(embeddings_list, axis=0)

        final_embeddings.append(mean_embedding)
        final_labels.append(employee_name)
        print(f"✅ Employee {employee_name}: Template generated from {len(embeddings_list)} images.")
    
    else:
        print(f"❌ Employee {employee_name}: No successful embeddings generated.")

# --- Step 5: Save the Biometric Database ---
if final_embeddings:
    # Convert lists to NumPy arrays
    X = np.array(final_embeddings)
    Y = np.array(final_labels)

    # **THIS IS THE CODE THAT SAVES THE EMBEDDINGS**
    np.savez_compressed(EMBEDDING_FILE, X, Y)
    
    print(f"\nSuccessfully saved {len(X)} employee templates to {EMBEDDING_FILE}")
else:
    print("\nNo employee templates were successfully generated.")