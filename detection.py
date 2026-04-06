"""
detection.py — Face Detection Module (Person A's responsibility)
Updated: Day 2

This module provides face detection using two CNN-based detectors:
  1. MTCNN  — Multi-task Cascaded Convolutional Network
  2. RetinaFace — High-precision single-stage detector

NEW in Day 2:
  - Image preprocessing pipeline (histogram equalisation, denoising, resizing)
  - Robust edge case handling (no face, tiny face, negative box coords)
  - Timing measurements for speed comparison
  - Both detectors fully implemented and tested
  - Visualisation and cropped-face saving helpers

Each detector returns a STANDARDISED output format so Person B's
recognition module works identically with either detector.

═══════════════════════════════════════════════════════════════
PERSON A — KEY CONCEPTS TO UNDERSTAND FOR YOUR REPORT & VIVA
═══════════════════════════════════════════════════════════════

1. MTCNN (Multi-task Cascaded Convolutional Networks):
   - Uses a 3-stage cascade architecture:
     Stage 1 — P-Net (Proposal Network):
       Creates an image pyramid (same image at many scales) and runs a
       shallow CNN across each scale to propose candidate face regions.
       Fast but produces many false positives.
     Stage 2 — R-Net (Refinement Network):
       Takes P-Net's proposals, runs a deeper CNN on each one, and rejects
       false positives. Also refines the bounding box coordinates.
     Stage 3 — O-Net (Output Network):
       Final stage — runs the deepest CNN, outputs accurate bounding boxes
       AND 5 facial landmarks (left_eye, right_eye, nose, mouth_left, mouth_right).
   - WHY CASCADE? Each stage filters out easy negatives early, so the expensive
     final stage only processes a few high-quality candidates.

2. RetinaFace:
   - Single-stage detector using a Feature Pyramid Network (FPN) backbone.
   - FPN creates feature maps at multiple scales from a single forward pass,
     allowing the network to detect faces of all sizes simultaneously.
   - Uses "anchor boxes" — predefined box templates at each position in the
     feature map. The network predicts offsets from these anchors.
   - Generally more accurate than MTCNN, especially on small/occluded faces.

3. Face Alignment:
   - A face tilted even 10-15° produces a different embedding vector.
   - We use eye landmarks to compute the tilt angle and rotate the image
     so both eyes sit on a horizontal line.

4. Image Preprocessing:
   - CLAHE histogram equalisation improves contrast in poor lighting.
   - Gaussian denoising reduces sensor noise from low-quality cameras.
   - Frame resizing standardises input and speeds up detection.
"""

import cv2
import numpy as np
import time
from mtcnn import MTCNN
import config


# ══════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════
class FacePreprocessor:
    """
    Preprocesses images before detection to improve accuracy.

    Handles three common problems in real-world office camera footage:
      1. Poor lighting → CLAHE histogram equalisation
      2. Camera noise  → Gaussian denoising
      3. Large images  → resizing to standard resolution

    WHY CLAHE INSTEAD OF REGULAR HISTOGRAM EQUALISATION?
      Regular equalisation applies globally — it can over-brighten already
      bright areas. CLAHE works on small tiles independently, preserving
      local contrast. The 'clip limit' prevents over-amplification of noise.
    """

    def __init__(self, target_width=config.FRAME_WIDTH, target_height=config.FRAME_HEIGHT):
        """
        Initialise the preprocessor.

        Parameters:
            target_width (int): Width to resize frames to.
            target_height (int): Height to resize frames to.
        """
        self.target_width = target_width
        self.target_height = target_height

        # CLAHE — Contrast Limited Adaptive Histogram Equalisation
        # clipLimit=2.0: limits contrast amplification to prevent noise boosting
        # tileGridSize=(8,8): divides the image into 8x8 tiles for local processing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize_frame(self, image):
        """
        Resize image to standard dimensions while preserving aspect ratio.

        A 640x480 image runs 4-8x faster through detection than 1920x1080.

        Parameters:
            image (numpy.ndarray): Input image (any size).

        Returns:
            tuple: (resized_image, scale_factor)
                   scale_factor lets us map boxes back to original coordinates.
        """
        h, w = image.shape[:2]

        if w <= self.target_width and h <= self.target_height:
            return image, 1.0

        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def equalise_histogram(self, image):
        """
        Apply CLAHE to the lightness channel in LAB colour space.

        Converting to LAB and equalising only L preserves colour fidelity
        while improving brightness/contrast.

        Parameters:
            image (numpy.ndarray): BGR image.

        Returns:
            numpy.ndarray: Contrast-enhanced BGR image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def denoise(self, image, kernel_size=5):
        """
        Apply Gaussian blur to reduce high-frequency noise.

        Parameters:
            image (numpy.ndarray): BGR image.
            kernel_size (int): Blur kernel size (must be odd). Default 5.

        Returns:
            numpy.ndarray: Denoised image.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def preprocess(self, image, apply_clahe=True, apply_denoise=True, apply_resize=True):
        """
        Run the full preprocessing pipeline.

        Parameters:
            image (numpy.ndarray): Raw BGR input image.
            apply_clahe (bool): Apply histogram equalisation.
            apply_denoise (bool): Apply denoising.
            apply_resize (bool): Resize to standard dimensions.

        Returns:
            tuple: (processed_image, scale_factor)
        """
        processed = image.copy()
        scale = 1.0

        if apply_resize:
            processed, scale = self.resize_frame(processed)
        if apply_clahe:
            processed = self.equalise_histogram(processed)
        if apply_denoise:
            processed = self.denoise(processed)

        return processed, scale


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def align_face(image, left_eye, right_eye):
    """
    Aligns a face so both eyes are on a horizontal line.

    Steps:
      1. Compute the angle between the two eye positions via arctan2.
      2. Find the midpoint between the eyes (rotation centre).
      3. Build a 2×3 affine rotation matrix.
      4. Warp the image to remove the tilt.

    WHY THIS MATTERS:
      Recognition models are trained on aligned faces. A 10° tilt can
      reduce cosine similarity by 0.1-0.2, crossing the threshold and
      causing false rejects.

    Parameters:
        image (numpy.ndarray): Full image (BGR).
        left_eye (tuple): (x, y) of the left eye centre.
        right_eye (tuple): (x, y) of the right eye centre.

    Returns:
        numpy.ndarray: Rotated image (same dimensions).
    """
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned = cv2.warpAffine(
        image, rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )
    return aligned


def crop_and_resize(image, box, target_size):
    """
    Crop a face from the image using the bounding box and resize it.

    Adds a 10% margin for forehead/chin inclusion. Handles edge cases
    like negative coordinates (faces at image edges) and tiny crops.

    Parameters:
        image (numpy.ndarray): Full image.
        box (tuple): (x, y, width, height) bounding box.
        target_size (tuple): (width, height) to resize to.

    Returns:
        numpy.ndarray or None: Cropped and resized face, or None if invalid.
    """
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img_h, img_w = image.shape[:2]

    if w <= 0 or h <= 0:
        print("[WARN] Invalid bounding box dimensions. Skipping.")
        return None

    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        print("[WARN] Face crop too small. Skipping.")
        return None

    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    return face_resized


def validate_detection(box, confidence, image_shape, min_confidence=0.5, min_face_px=30):
    """
    Validates a detection result by checking confidence, size, and visibility.

    Parameters:
        box (tuple): (x, y, w, h).
        confidence (float): Detector confidence (0 to 1).
        image_shape (tuple): (height, width, channels).
        min_confidence (float): Minimum acceptable confidence.
        min_face_px (int): Minimum face size in pixels.

    Returns:
        bool: True if detection passes all checks.
    """
    x, y, w, h = box
    img_h, img_w = image_shape[:2]

    if confidence < min_confidence:
        return False
    if w < min_face_px or h < min_face_px:
        return False

    # At least 50% of face must be within the image
    visible_x = min(x + w, img_w) - max(x, 0)
    visible_y = min(y + h, img_h) - max(y, 0)
    visible_area = max(0, visible_x) * max(0, visible_y)
    total_area = w * h

    if total_area <= 0 or (visible_area / total_area) < 0.5:
        return False

    return True


# ══════════════════════════════════════════════════════════════
#  MTCNN DETECTOR
# ══════════════════════════════════════════════════════════════
class MTCNNDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).

    3-stage cascade:
      P-Net (12×12) → proposes regions, very fast
      R-Net (24×24) → refines, rejects ~70% false positives
      O-Net (48×48) → final boxes + 5 landmarks

    Each stage simultaneously predicts:
      1. Face/not-face classification
      2. Bounding box regression
      3. Landmark localisation (O-Net only)
    """

    def __init__(self):
        self.detector = MTCNN(
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            steps_threshold=config.MTCNN_THRESHOLDS,
            scale_factor=config.MTCNN_SCALE_FACTOR
        )
        self.preprocessor = FacePreprocessor()
        print("[INFO] MTCNN detector initialised.")

    def detect_faces(self, image, preprocess=True):
        """
        Detect all faces in an image using MTCNN.

        Parameters:
            image (numpy.ndarray): Input BGR image.
            preprocess (bool): Apply preprocessing pipeline.

        Returns:
            list of dict with keys: box, confidence, landmarks,
            face_cropped (160×160), face_cropped_arcface (112×112),
            face_aligned, detector, detection_time_ms
        """
        start_time = time.time()

        if preprocess:
            processed, scale = self.preprocessor.preprocess(image)
        else:
            processed = image.copy()
            scale = 1.0

        # MTCNN expects RGB
        image_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(image_rgb)

        detection_time = (time.time() - start_time) * 1000

        results = []
        for det in detections:
            box = det['box']
            confidence = det['confidence']
            keypoints = det['keypoints']

            # Scale back to original image coordinates
            if scale != 1.0:
                box = [int(b / scale) for b in box]
                keypoints = {
                    k: (int(v[0] / scale), int(v[1] / scale))
                    for k, v in keypoints.items()
                }

            if not validate_detection(box, confidence, image.shape):
                continue

            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            aligned_image = align_face(image, left_eye, right_eye)
            face_facenet = crop_and_resize(aligned_image, box, config.FACENET_INPUT_SIZE)
            face_arcface = crop_and_resize(aligned_image, box, config.ARCFACE_INPUT_SIZE)

            if face_facenet is None or face_arcface is None:
                continue

            results.append({
                'box': tuple(box),
                'confidence': confidence,
                'landmarks': keypoints,
                'face_cropped': face_facenet,
                'face_cropped_arcface': face_arcface,
                'face_aligned': aligned_image,
                'detector': 'mtcnn',
                'detection_time_ms': detection_time
            })

        print(f"[MTCNN] Detected {len(results)} face(s) in {detection_time:.1f}ms.")
        return results


# ══════════════════════════════════════════════════════════════
#  RETINAFACE DETECTOR
# ══════════════════════════════════════════════════════════════
class RetinaFaceDetector:
    """
    Face detector using RetinaFace.

    Single-stage detector with FPN backbone:
      - Backbone (ResNet/MobileNet) extracts hierarchical features
      - FPN merges multi-scale features for detecting all face sizes
      - Context module uses deformable convolutions for surroundings
      - Multi-task heads predict: class, box, landmarks simultaneously

    TRADE-OFFS vs MTCNN:
      + More accurate on small/occluded/angled faces
      + Better on WIDER FACE benchmarks
      - Slower (heavier backbone)
      - Uses more memory
    """

    def __init__(self):
        try:
            from retinaface import RetinaFace as RF
            self.detector = RF
            self.preprocessor = FacePreprocessor()
            self.available = True
            print("[INFO] RetinaFace detector initialised.")
        except ImportError:
            print("[WARN] RetinaFace not installed. Run: pip install retina-face")
            self.detector = None
            self.available = False

    def detect_faces(self, image, preprocess=True):
        """
        Detect all faces using RetinaFace.

        Returns SAME output format as MTCNNDetector.detect_faces() so
        Person B's recognition module works with either detector unchanged.

        Parameters:
            image (numpy.ndarray): Input BGR image.
            preprocess (bool): Apply preprocessing pipeline.

        Returns:
            list of dict (same structure as MTCNN output).
        """
        if not self.available:
            print("[ERROR] RetinaFace not available.")
            return []

        start_time = time.time()

        if preprocess:
            processed, scale = self.preprocessor.preprocess(image)
        else:
            processed = image.copy()
            scale = 1.0

        detections = self.detector.detect_faces(processed)
        detection_time = (time.time() - start_time) * 1000

        results = []
        if detections is None:
            print(f"[RetinaFace] No faces detected ({detection_time:.1f}ms).")
            return results

        for key, det in detections.items():
            facial_area = det['facial_area']    # [x1, y1, x2, y2]
            landmarks = det['landmarks']
            score = det['score']

            # Convert [x1, y1, x2, y2] → [x, y, w, h]
            x1, y1, x2, y2 = facial_area
            box = [x1, y1, x2 - x1, y2 - y1]

            # Scale back to original coordinates
            if scale != 1.0:
                box = [int(b / scale) for b in box]
                landmarks = {
                    k: (int(v[0] / scale), int(v[1] / scale))
                    for k, v in landmarks.items()
                }
            else:
                landmarks = {
                    k: (int(v[0]), int(v[1]))
                    for k, v in landmarks.items()
                }

            if not validate_detection(box, score, image.shape,
                                      min_confidence=config.RETINA_CONFIDENCE_THRESHOLD):
                continue

            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']

            aligned_image = align_face(image, left_eye, right_eye)
            face_facenet = crop_and_resize(aligned_image, box, config.FACENET_INPUT_SIZE)
            face_arcface = crop_and_resize(aligned_image, box, config.ARCFACE_INPUT_SIZE)

            if face_facenet is None or face_arcface is None:
                continue

            keypoints = {
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose': landmarks['nose'],
                'mouth_left': landmarks['mouth_left'],
                'mouth_right': landmarks['mouth_right']
            }

            results.append({
                'box': tuple(box),
                'confidence': score,
                'landmarks': keypoints,
                'face_cropped': face_facenet,
                'face_cropped_arcface': face_arcface,
                'face_aligned': aligned_image,
                'detector': 'retinaface',
                'detection_time_ms': detection_time
            })

        print(f"[RetinaFace] Detected {len(results)} face(s) in {detection_time:.1f}ms.")
        return results


# ══════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ══════════════════════════════════════════════════════════════
def get_detector(model_name=None):
    """Returns detector instance by name (default from config)."""
    name = model_name or config.DETECTION_MODEL

    if name.lower() == "mtcnn":
        return MTCNNDetector()
    elif name.lower() == "retinaface":
        return RetinaFaceDetector()
    else:
        raise ValueError(f"Unknown detector: {name}. Use 'mtcnn' or 'retinaface'.")


def get_all_detectors():
    """
    Returns both detectors for comparison testing.

    Returns:
        dict: {"mtcnn": MTCNNDetector, "retinaface": RetinaFaceDetector or None}
    """
    detectors = {"mtcnn": MTCNNDetector()}
    try:
        retina = RetinaFaceDetector()
        detectors["retinaface"] = retina if retina.available else None
    except Exception:
        detectors["retinaface"] = None
    return detectors


# ══════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════
def draw_detections(image, detections, colour=(0, 255, 0), label_prefix=""):
    """
    Draw bounding boxes and landmarks on an image for debugging.

    Parameters:
        image (numpy.ndarray): Image to annotate (will be copied).
        detections (list): Output from detect_faces().
        colour (tuple): BGR colour for boxes.
        label_prefix (str): Label prefix, e.g. "MTCNN".

    Returns:
        numpy.ndarray: Annotated copy.
    """
    annotated = image.copy()

    for i, det in enumerate(detections):
        x, y, w, h = [int(v) for v in det['box']]

        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)

        label = f"{label_prefix} {det['confidence']:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated,
                      (x, y - text_size[1] - 8),
                      (x + text_size[0] + 4, y),
                      colour, -1)
        cv2.putText(annotated, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        landmark_colours = {
            'left_eye': (255, 0, 0),
            'right_eye': (255, 0, 0),
            'nose': (0, 255, 255),
            'mouth_left': (0, 0, 255),
            'mouth_right': (0, 0, 255)
        }
        for lm_name, point in det['landmarks'].items():
            lm_colour = landmark_colours.get(lm_name, (255, 255, 255))
            cv2.circle(annotated, (int(point[0]), int(point[1])), 3, lm_colour, -1)

    return annotated


def save_cropped_faces(detections, output_dir, prefix="face"):
    """
    Save all cropped faces to disk for inspection.

    Returns:
        list of str: Paths to saved images.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, det in enumerate(detections):
        fn_path = os.path.join(output_dir, f"{prefix}_{i+1}_facenet_160x160.jpg")
        cv2.imwrite(fn_path, det['face_cropped'])
        paths.append(fn_path)

        af_path = os.path.join(output_dir, f"{prefix}_{i+1}_arcface_112x112.jpg")
        cv2.imwrite(af_path, det['face_cropped_arcface'])
        paths.append(af_path)

    return paths


# ══════════════════════════════════════════════════════════════
#  QUICK TEST — run this file directly
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys, os

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Could not load: {img_path}")
            sys.exit(1)
        print(f"[INFO] Loaded: {img_path} ({image.shape[1]}x{image.shape[0]})")
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("[ERROR] Webcam failed. Provide an image path instead.")
            sys.exit(1)
        print("[INFO] Webcam frame captured.")

    print("\n--- Testing MTCNN ---")
    mtcnn = MTCNNDetector()
    mtcnn_results = mtcnn.detect_faces(image)
    for i, f in enumerate(mtcnn_results):
        print(f"  Face {i+1}: conf={f['confidence']:.4f}, box={f['box']}, "
              f"time={f['detection_time_ms']:.1f}ms")

    print("\n--- Testing RetinaFace ---")
    retina = RetinaFaceDetector()
    if retina.available:
        retina_results = retina.detect_faces(image)
        for i, f in enumerate(retina_results):
            print(f"  Face {i+1}: conf={f['confidence']:.4f}, box={f['box']}, "
                  f"time={f['detection_time_ms']:.1f}ms")

    output_dir = os.path.join(config.DATA_DIR, "day2_crops")
    if mtcnn_results:
        save_cropped_faces(mtcnn_results, output_dir, "mtcnn")
        print(f"\n[INFO] Cropped faces saved to {output_dir}")

    print("\n[DONE] Day 2 detection test complete.")
