import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "sample_images")
DATABASE_PATH = os.path.join(DATA_DIR, "known_faces.json")

for d in [DATA_DIR, KNOWN_FACES_DIR, SAMPLE_IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# MTCNN detection parameters
MTCNN_MIN_FACE_SIZE = 40
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]   # P-Net, R-Net, O-Net confidence thresholds
MTCNN_SCALE_FACTOR = 0.709

# RetinaFace detection parameters
RETINA_CONFIDENCE_THRESHOLD = 0.9

# Input sizes expected by each recognition model
FACENET_INPUT_SIZE = (160, 160)
ARCFACE_INPUT_SIZE = (112, 112)

# Recognition thresholds (cosine similarity)
FACENET_THRESHOLD = 0.6
ARCFACE_THRESHOLD = 0.4

# Active models — change these to switch between models
DETECTION_MODEL = "mtcnn"       # "mtcnn" or "retinaface"
RECOGNITION_MODEL = "facenet"   # "facenet" or "arcface"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_INTERVAL = 2
