"""
recognition.py — Face Recognition Module (Person B's responsibility)

This module will handle:
  1. Loading pre-trained models (FaceNet, ArcFace)
  2. Extracting face embeddings from cropped face images
  3. Comparing embeddings using cosine similarity
  4. Returning the best match or "unknown"

TODO (Person B — Day 2):
  - Implement FaceNetRecogniser class
  - Implement embedding extraction
  - Implement cosine similarity matching

TODO (Person B — Day 3):
  - Implement ArcFaceRecogniser class
  - Add threshold-based matching logic
  - Add get_recogniser() factory function
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config


class FaceNetRecogniser:
    """
    Face recognition using FaceNet (128-dimensional embeddings).

    FaceNet maps face images to a compact 128D vector space where:
      - Faces of the same person are close together (low distance)
      - Faces of different people are far apart (high distance)

    The model was trained using a 'triplet loss' function:
      - Anchor: a face image of person X
      - Positive: another image of person X (should be close)
      - Negative: an image of person Y (should be far)

    TODO: Person B implements this on Day 2.
    """

    def __init__(self):
        # TODO: Load the FaceNet model using facenet-pytorch
        # from facenet_pytorch import InceptionResnetV1
        # self.model = InceptionResnetV1(pretrained='vggface2').eval()
        print("[INFO] FaceNetRecogniser — placeholder (implement on Day 2)")

    def extract_embedding(self, face_image):
        """
        Convert a cropped 160x160 face image into a 128D embedding vector.

        Parameters:
            face_image (numpy.ndarray): Cropped face, 160x160, BGR.

        Returns:
            numpy.ndarray: 128-dimensional embedding vector.
        """
        # TODO: Implement
        # 1. Convert BGR to RGB
        # 2. Convert to tensor, normalise to [-1, 1]
        # 3. Run through self.model
        # 4. Return the embedding as numpy array
        pass

    def compare(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.

        Returns:
            float: Similarity score (1.0 = identical, 0.0 = completely different)
        """
        return cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]


class ArcFaceRecogniser:
    """
    Face recognition using ArcFace (512-dimensional embeddings).

    ArcFace uses an 'Additive Angular Margin Loss' which forces the model
    to learn even more discriminative features than FaceNet's triplet loss.

    TODO: Person B implements this on Day 3.
    """

    def __init__(self):
        # TODO: Load ArcFace model
        print("[INFO] ArcFaceRecogniser — placeholder (implement on Day 3)")

    def extract_embedding(self, face_image):
        """Extract a 512D embedding from a 112x112 face image."""
        # TODO: Implement
        pass


def get_recogniser(model_name=None):
    """Factory function to get the appropriate recogniser."""
    name = model_name or config.RECOGNITION_MODEL
    if name.lower() == "facenet":
        return FaceNetRecogniser()
    elif name.lower() == "arcface":
        return ArcFaceRecogniser()
    else:
        raise ValueError(f"Unknown recogniser: {name}")
