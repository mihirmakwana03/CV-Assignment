"""
recognition.py - Face Recognition Module
Extracts face embeddings and matches them against known identities.
Supports FaceNet (128D) and ArcFace (512D).

Person B 
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
import cv2
import config


class FaceNetRecogniser:
    """
    FaceNet maps a 160x160 face image to a 128-dimensional embedding vector.
    """

    def __init__(self):

        print("[FaceNet] Loading model...")
        self.model = InceptionResnetV1(pretrained='vggface2').eval()

    def extract_embedding(self, face_image):

        # Convert BGR → RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Normalize image
        face_image = face_image.astype(np.float32) / 255.0

        # Convert to tensor shape
        face_image = np.transpose(face_image, (2, 0, 1))
        face_image = np.expand_dims(face_image, axis=0)

        embedding = self.model(face_image)

        return embedding.detach().numpy()[0]

    def compare(self, emb1, emb2):

        return cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]


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