"""
Configuration settings for Smart Office Face Recognition System.
All tuneable parameters are centralised here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")      # saved face images
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")   # test images
DATABASE_PATH = os.path.join(DATA_DIR, "known_faces.json")    # persistent storage

# Create directories if they don't exist
for d in [DATA_DIR, KNOWN_FACES_DIR, SAMPLE_IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Face Detection Settings
# ──────────────────────────────────────────────
# MTCNN settings
MTCNN_MIN_FACE_SIZE = 40          # minimum face size in pixels
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]  # confidence thresholds for 3 MTCNN stages
MTCNN_SCALE_FACTOR = 0.709        # image pyramid scale factor

# RetinaFace settings
RETINA_CONFIDENCE_THRESHOLD = 0.9

# ──────────────────────────────────────────────
# Face Preprocessing
# ──────────────────────────────────────────────
FACENET_INPUT_SIZE = (160, 160)    # FaceNet expects 160x160
ARCFACE_INPUT_SIZE = (112, 112)    # ArcFace expects 112x112

# ──────────────────────────────────────────────
# Face Recognition Settings
# ──────────────────────────────────────────────
# Similarity thresholds (cosine distance)
# Lower = stricter matching, Higher = more lenient
FACENET_THRESHOLD = 0.6
ARCFACE_THRESHOLD = 0.4

# Which models to use (can switch for comparison)
DETECTION_MODEL = "mtcnn"          # "mtcnn" or "retinaface"
RECOGNITION_MODEL = "facenet"      # "facenet" or "arcface"

# ──────────────────────────────────────────────
# Camera / Input Settings
# ──────────────────────────────────────────────
CAMERA_INDEX = 0                   # default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_INTERVAL = 2             # seconds between detection attempts (for slower CPUs)
