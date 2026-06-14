import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# Step 2 : 
def face_embeddings(img_path):
    # step 1 : 
    mtcnn = MTCNN(
        image_size = 160, margin=20, min_face_size=20, thresholds=[0.6, 0.7, 0.7]
        , factor=0.709, post_process=True).eval() 
    # using eval to set my model for evaluation mode

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    try:
        img = Image.open(img_path).convert('RGB')

        # detect and extract faces using embedding model
        face_tensor = mtcnn(img)

        if face_tensor is None:
            return None
        
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0))
        
        return embedding.numpy().flatten()
    except Exception as e:
        print(f"[ERROR] Processing {img_path}: {e}")
        return None