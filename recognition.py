"""
recognition.py - Face Recognition Module
Extracts face embeddings and matches them against known identities.
Supports FaceNet (128D) and ArcFace (512D).

Person B is responsible for this module.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config


class FaceNetRecogniser:
    """
    FaceNet maps a 160x160 face image to a 128-dimensional embedding vector.
    Faces of the same person cluster together in this vector space.
    
    Person B: implement extract_embedding() using facenet_pytorch.InceptionResnetV1
    """

    def __init__(self):
        # TODO: load model
        # from facenet_pytorch import InceptionResnetV1
        # self.model = InceptionResnetV1(pretrained='vggface2').eval()
        print("[FaceNet] Placeholder - Person B to implement")

    def extract_embedding(self, face_image):
        """
        Takes a 160x160 BGR face image (from detection.detect() -> face_160),
        returns a 128D numpy array.
        """
        # TODO: convert BGR->RGB, normalise to tensor, run model, return numpy
        pass

    def compare(self, emb1, emb2):
        """Cosine similarity between two embeddings. 1.0 = identical."""
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]


class ArcFaceRecogniser:
    """
    ArcFace produces 512D embeddings with better discriminative power than FaceNet.
    
    Person B: implement using insightface or deepface wrapper.
    """

    def __init__(self):
        # TODO: load model
        print("[ArcFace] Placeholder - Person B to implement")

    def extract_embedding(self, face_image):
        """
        Takes a 112x112 BGR face image (from detection.detect() -> face_112),
        returns a 512D numpy array.
        """
        # TODO: implement
        pass


def get_recogniser(name=None):
    """Get recogniser by name. Defaults to config setting."""
    name = (name or config.RECOGNITION_MODEL).lower()
    if name == "facenet":
        return FaceNetRecogniser()
    elif name == "arcface":
        return ArcFaceRecogniser()
    raise ValueError(f"Unknown recogniser: {name}")
