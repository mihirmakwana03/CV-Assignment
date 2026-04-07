"""
recognition.py - Face Recognition Module
Extracts face embeddings and matches them against known identities.
Supports FaceNet (128D) and ArcFace (512D).

Person B 
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
import config

class FaceNetRecogniser:
    """
    FaceNet maps a 160x160 face image to a 128-dimensional embedding vector.
    """

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FaceNet] Loading model on {self.device}...")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def extract_embedding(self, face_image):

        # face_image can be np.ndarray or torch.Tensor
        if isinstance(face_image, np.ndarray):
            x = torch.from_numpy(face_image)
        elif torch.is_tensor(face_image):
            x = face_image
        else:
            raise TypeError(f"Unsupported face_image type: {type(face_image)}")

        x = x.float()

        # HWC -> CHW
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)

        # add batch dim
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # normalize if coming as 0..255
        if x.max() > 1.5:
            x = x / 255.0

        # FaceNet (facenet-pytorch) commonly uses [-1, 1]
        x = (x - 0.5) / 0.5

        x = x.to(self.device)

        with torch.no_grad():
            emb = self.model(x)

        emb = emb.detach().cpu().numpy().reshape(-1)   # force 1D vector
        return emb

    def compare(self, emb1, emb2):

        e1 = np.asarray(emb1, dtype=np.float32).reshape(-1)
        e2 = np.asarray(emb2, dtype=np.float32).reshape(-1)

        if e1.shape[0] != e2.shape[0]:
            # skip incompatible records
            return -1.0

        return float(cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0, 0])


class ArcFaceRecogniser:

    """
    ArcFace produces 512D embeddings
    """

    def __init__(self):

        print("[ArcFace] Ready (DeepFace backend)")

    def extract_embedding(self, face_image):

        embedding = DeepFace.represent(
            img_path=face_image,
            model_name="ArcFace",
            enforce_detection=False
        )

        return np.array(embedding[0]["embedding"])


def get_recogniser(name=None):

    name = (name or config.RECOGNITION_MODEL).lower()

    if name == "facenet":
        return FaceNetRecogniser()

    elif name == "arcface":
        return ArcFaceRecogniser()

    raise ValueError(f"Unknown recogniser: {name}")