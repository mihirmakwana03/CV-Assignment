"""
detection.py - Face Detection Module
Handles face detection, alignment, cropping and preprocessing.
Supports MTCNN (PyTorch) and RetinaFace as detection backends.

Person A .
"""

import cv2
import numpy as np
import time
import torch
from facenet_pytorch import MTCNN as PyTorchMTCNN
from PIL import Image
import config


# ---------------------------------------------------------------------------
#  Preprocessing - improves detection on low-quality or poorly lit images
# ---------------------------------------------------------------------------

class Preprocessor:
    """Handles image preprocessing before face detection."""

    def __init__(self):
        # CLAHE operates on small tiles rather than the whole image,
        # so it enhances local contrast without blowing out bright areas
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize(self, image, max_w=config.FRAME_WIDTH, max_h=config.FRAME_HEIGHT):
        """Downscale large images to speed up detection. Returns (image, scale)."""
        h, w = image.shape[:2]
        if w <= max_w and h <= max_h:
            return image, 1.0
        scale = min(max_w / w, max_h / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def enhance_contrast(self, image):
        """Apply CLAHE on the lightness channel (LAB space) to fix poor lighting."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def denoise(self, image):
        """Light Gaussian blur to reduce camera sensor noise."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def run(self, image):
        """Full pipeline: resize -> contrast -> denoise. Returns (processed, scale)."""
        processed, scale = self.resize(image)
        processed = self.enhance_contrast(processed)
        processed = self.denoise(processed)
        return processed, scale


# ---------------------------------------------------------------------------
#  Utility functions used by both detectors
# ---------------------------------------------------------------------------

def align_face(image, left_eye, right_eye):
    """
    Rotate the image so both eyes sit on a horizontal line.
    This normalises head tilt which would otherwise throw off
    the embedding vectors during recognition.
    """
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) // 2,
              (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)


def crop_face(image, box, target_size):
    """
    Crop a face region from the image with a small margin,
    then resize to target_size. Returns None if the crop is invalid.
    """
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if w <= 0 or h <= 0:
        return None

    img_h, img_w = image.shape[:2]
    # 10% margin so we don't chop off foreheads/chins
    mx, my = int(w * 0.1), int(h * 0.1)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return cv2.resize(image[y1:y2, x1:x2], target_size, interpolation=cv2.INTER_AREA)


def is_valid_detection(box, confidence, img_shape, min_conf=0.5, min_size=30):
    """Filter out weak detections: low confidence, tiny faces, or mostly out of frame."""
    x, y, w, h = box
    if confidence < min_conf or w < min_size or h < min_size:
        return False
    # check at least half the face is inside the image
    img_h, img_w = img_shape[:2]
    vis_w = min(x + w, img_w) - max(x, 0)
    vis_h = min(y + h, img_h) - max(y, 0)
    if w * h <= 0:
        return False
    return (max(0, vis_w) * max(0, vis_h)) / (w * h) >= 0.5


def _scale_coords(val, scale):
    """Helper to map coordinates back to original image size after preprocessing."""
    if scale == 1.0:
        return val
    return int(val / scale)


# ---------------------------------------------------------------------------
#  MTCNN Detector (uses facenet-pytorch, NOT the tensorflow-based mtcnn package)
# ---------------------------------------------------------------------------

class MTCNNDetector:
    """
    MTCNN is a 3-stage cascade detector:
      P-Net -> proposes candidate face regions
      R-Net -> refines proposals, rejects false positives
      O-Net -> outputs final bounding boxes + 5 facial landmarks
    
    We use the PyTorch implementation from facenet-pytorch because
    the original mtcnn pip package depends on TensorFlow which causes
    compatibility issues on most setups.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PyTorchMTCNN(
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            thresholds=config.MTCNN_THRESHOLDS,
            factor=config.MTCNN_SCALE_FACTOR,
            select_largest=False,   # return all faces, not just the biggest
            post_process=False,
            device=self.device
        )
        self.preprocessor = Preprocessor()
        print(f"[MTCNN] Ready (device: {self.device})")

    def detect(self, image, use_preprocessing=True):
        """
        Detect faces in a BGR image.
        Returns a list of dicts, each with:
            box, confidence, landmarks, face_160, face_112, detector, time_ms
        """
        t0 = time.time()

        if use_preprocessing:
            processed, scale = self.preprocessor.run(image)
        else:
            processed, scale = image.copy(), 1.0

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        boxes, probs, points = self.model.detect(pil_img, landmarks=True)
        elapsed = (time.time() - t0) * 1000

        if boxes is None:
            return []

        results = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(probs[i])
            lm = points[i]  # shape (5, 2): left_eye, right_eye, nose, mouth_l, mouth_r

            # convert x1,y1,x2,y2 -> x,y,w,h and scale to original image coords
            box = [
                _scale_coords(x1, scale), _scale_coords(y1, scale),
                _scale_coords(x2 - x1, scale), _scale_coords(y2 - y1, scale)
            ]

            landmarks = {
                'left_eye':    (_scale_coords(lm[0][0], scale), _scale_coords(lm[0][1], scale)),
                'right_eye':   (_scale_coords(lm[1][0], scale), _scale_coords(lm[1][1], scale)),
                'nose':        (_scale_coords(lm[2][0], scale), _scale_coords(lm[2][1], scale)),
                'mouth_left':  (_scale_coords(lm[3][0], scale), _scale_coords(lm[3][1], scale)),
                'mouth_right': (_scale_coords(lm[4][0], scale), _scale_coords(lm[4][1], scale)),
            }

            if not is_valid_detection(box, conf, image.shape):
                continue

            aligned = align_face(image, landmarks['left_eye'], landmarks['right_eye'])
            face_160 = crop_face(aligned, box, config.FACENET_INPUT_SIZE)
            face_112 = crop_face(aligned, box, config.ARCFACE_INPUT_SIZE)

            if face_160 is None or face_112 is None:
                continue

            results.append({
                'box': tuple(box),
                'confidence': conf,
                'landmarks': landmarks,
                'face_160': face_160,       # ready for FaceNet
                'face_112': face_112,       # ready for ArcFace
                'detector': 'mtcnn',
                'time_ms': elapsed
            })

        return results


# ---------------------------------------------------------------------------
#  RetinaFace Detector
# ---------------------------------------------------------------------------

class RetinaFaceDetector:
    """
    RetinaFace uses a Feature Pyramid Network backbone to detect faces
    at multiple scales in a single forward pass. More accurate than MTCNN
    on small or partially occluded faces, but slower.
    
    Install with: pip install retina-face --no-deps
    """

    def __init__(self):
        self.available = False
        try:
            from retinaface import RetinaFace as RF
            self.model = RF
            self.available = True
            self.preprocessor = Preprocessor()
            print("[RetinaFace] Ready")
        except ImportError:
            print("[RetinaFace] Not installed - run: pip install retina-face --no-deps")

    def detect(self, image, use_preprocessing=True):
        """Same return format as MTCNNDetector.detect()"""
        if not self.available:
            return []

        t0 = time.time()

        if use_preprocessing:
            processed, scale = self.preprocessor.run(image)
        else:
            processed, scale = image.copy(), 1.0

        raw = self.model.detect_faces(processed)
        elapsed = (time.time() - t0) * 1000

        if raw is None:
            return []

        results = []
        for key, det in raw.items():
            x1, y1, x2, y2 = det['facial_area']
            score = det['score']
            lm = det['landmarks']

            box = [
                _scale_coords(x1, scale), _scale_coords(y1, scale),
                _scale_coords(x2 - x1, scale), _scale_coords(y2 - y1, scale)
            ]

            landmarks = {}
            for name in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
                landmarks[name] = (_scale_coords(lm[name][0], scale),
                                   _scale_coords(lm[name][1], scale))

            if not is_valid_detection(box, score, image.shape,
                                      min_conf=config.RETINA_CONFIDENCE_THRESHOLD):
                continue

            aligned = align_face(image, landmarks['left_eye'], landmarks['right_eye'])
            face_160 = crop_face(aligned, box, config.FACENET_INPUT_SIZE)
            face_112 = crop_face(aligned, box, config.ARCFACE_INPUT_SIZE)

            if face_160 is None or face_112 is None:
                continue

            results.append({
                'box': tuple(box),
                'confidence': score,
                'landmarks': landmarks,
                'face_160': face_160,
                'face_112': face_112,
                'detector': 'retinaface',
                'time_ms': elapsed
            })

        return results


# ---------------------------------------------------------------------------
#  Public API - other modules import these
# ---------------------------------------------------------------------------

def get_detector(name=None):
    """Get a detector by name. Defaults to whatever is set in config."""
    name = (name or config.DETECTION_MODEL).lower()
    if name == "mtcnn":
        return MTCNNDetector()
    elif name == "retinaface":
        return RetinaFaceDetector()
    raise ValueError(f"Unknown detector: {name}")


def get_all_detectors():
    """Returns dict of all available detectors for comparison testing."""
    detectors = {"mtcnn": MTCNNDetector()}
    r = RetinaFaceDetector()
    detectors["retinaface"] = r if r.available else None
    return detectors
