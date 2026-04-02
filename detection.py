"""
detection.py — Face Detection Module (Person A's responsibility)

This module provides face detection using two CNN-based detectors:
  1. MTCNN  — Multi-task Cascaded Convolutional Network
  2. RetinaFace — High-precision single-stage detector

Each detector returns:
  - Bounding boxes around detected faces
  - Facial landmarks (eyes, nose, mouth corners)
  - Cropped, aligned, and resized face images ready for embedding extraction

WHY MTCNN?
  MTCNN uses a 3-stage cascade (P-Net → R-Net → O-Net):
    Stage 1 (P-Net): Scans the image at multiple scales to propose candidate face regions.
    Stage 2 (R-Net): Refines the proposals and rejects false positives.
    Stage 3 (O-Net): Final refinement + outputs 5 facial landmarks.
  This cascade design balances speed and accuracy.

WHY RetinaFace?
  RetinaFace is a single-stage detector that uses a Feature Pyramid Network (FPN)
  to detect faces at multiple scales in one pass. It achieves higher accuracy than
  MTCNN, especially for small or partially occluded faces.
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import config


# ──────────────────────────────────────────────
# Helper: Face Alignment using Eye Coordinates
# ──────────────────────────────────────────────
def align_face(image, left_eye, right_eye):
    """
    Aligns a face so that both eyes are on a horizontal line.

    How it works:
      1. Calculate the angle between the two eye positions.
      2. Compute the center point between the eyes.
      3. Rotate the entire image around that center to make eyes level.

    This is important because tilted faces produce inconsistent embeddings.
    A face tilted 15° could be misidentified — alignment fixes this.

    Parameters:
        image (numpy.ndarray): The full image (BGR format from OpenCV).
        left_eye (tuple): (x, y) coordinates of the left eye.
        right_eye (tuple): (x, y) coordinates of the right eye.

    Returns:
        numpy.ndarray: The rotated/aligned image.
    """
    # Step 1: Calculate the angle between the eyes
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Step 2: Find the center point between both eyes
    center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    # Step 3: Create a rotation matrix and apply it
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned = cv2.warpAffine(
        image, rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    return aligned


# ──────────────────────────────────────────────
# Helper: Crop and Resize a Detected Face
# ──────────────────────────────────────────────
def crop_and_resize(image, box, target_size):
    """
    Crops a face from the image using the bounding box and resizes it.

    Adds a small margin (10%) around the face to avoid cutting off
    foreheads or chins, which would hurt recognition accuracy.

    Parameters:
        image (numpy.ndarray): The full image.
        box (tuple): (x, y, width, height) bounding box from detector.
        target_size (tuple): (width, height) to resize to, e.g. (160, 160).

    Returns:
        numpy.ndarray: Cropped and resized face image.
    """
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Add 10% margin around the face
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)

    # Clamp to image boundaries so we don't go out of bounds
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)

    # Crop and resize
    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

    return face_resized


# ══════════════════════════════════════════════
#  MTCNN Detector
# ══════════════════════════════════════════════
class MTCNNDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).

    MTCNN processes an image in 3 stages:
      - P-Net: Proposes candidate face regions (fast, rough).
      - R-Net: Refines candidates and removes false positives.
      - O-Net: Final bounding box + 5 facial landmarks.

    The 5 landmarks are: left_eye, right_eye, nose, mouth_left, mouth_right.
    We use the eye landmarks for face alignment.
    """

    def __init__(self):
        """Initialise the MTCNN detector with settings from config."""
        self.detector = MTCNN(
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            steps_threshold=config.MTCNN_THRESHOLDS,
            scale_factor=config.MTCNN_SCALE_FACTOR
        )
        print("[INFO] MTCNN detector initialised.")

    def detect_faces(self, image):
        """
        Detect all faces in an image.

        Parameters:
            image (numpy.ndarray): Input image in BGR format (from OpenCV).

        Returns:
            list of dict, each containing:
                - 'box': (x, y, w, h) bounding box
                - 'confidence': detection confidence score (0 to 1)
                - 'landmarks': dict with keys 'left_eye', 'right_eye', 'nose',
                               'mouth_left', 'mouth_right' — each is (x, y)
                - 'face_cropped': cropped & resized face for FaceNet (160x160)
                - 'face_cropped_arcface': cropped & resized face for ArcFace (112x112)
                - 'face_aligned': the full image after alignment (for debugging)
        """
        # MTCNN expects RGB, but OpenCV loads as BGR — so convert
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        detections = self.detector.detect_faces(image_rgb)

        results = []
        for det in detections:
            box = det['box']                 # [x, y, width, height]
            confidence = det['confidence']
            keypoints = det['keypoints']     # dict of landmark positions

            # Extract eye positions for alignment
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            # Step 1: Align the face using eye coordinates
            aligned_image = align_face(image, left_eye, right_eye)

            # Step 2: Crop and resize for FaceNet (160x160)
            face_facenet = crop_and_resize(
                aligned_image, box, config.FACENET_INPUT_SIZE
            )

            # Step 3: Crop and resize for ArcFace (112x112)
            face_arcface = crop_and_resize(
                aligned_image, box, config.ARCFACE_INPUT_SIZE
            )

            results.append({
                'box': tuple(box),
                'confidence': confidence,
                'landmarks': keypoints,
                'face_cropped': face_facenet,
                'face_cropped_arcface': face_arcface,
                'face_aligned': aligned_image
            })

        print(f"[MTCNN] Detected {len(results)} face(s).")
        return results


# ══════════════════════════════════════════════
#  RetinaFace Detector (Person A adds this on Day 2)
# ══════════════════════════════════════════════
class RetinaFaceDetector:
    """
    Face detector using RetinaFace.

    RetinaFace uses a Feature Pyramid Network (FPN) backbone to detect
    faces at multiple scales in a single forward pass. It outputs:
      - Bounding boxes
      - 5 facial landmarks
      - Confidence scores

    Generally more accurate than MTCNN, especially on:
      - Small faces (far from camera)
      - Partially occluded faces
      - Extreme angles

    TODO (Person A — Day 2): Implement this detector.
    """

    def __init__(self):
        """Initialise RetinaFace detector."""
        try:
            from retinaface import RetinaFace as RF
            self.detector = RF
            print("[INFO] RetinaFace detector initialised.")
        except ImportError:
            print("[ERROR] RetinaFace not installed. Run: pip install retina-face")
            self.detector = None

    def detect_faces(self, image):
        """
        Detect all faces using RetinaFace.

        Parameters:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            list of dict with same structure as MTCNNDetector.detect_faces()
        """
        if self.detector is None:
            print("[ERROR] RetinaFace not available.")
            return []

        # RetinaFace works on file paths or numpy arrays
        detections = self.detector.detect_faces(image)

        results = []
        if detections is None:
            return results

        for key, det in detections.items():
            facial_area = det['facial_area']  # [x1, y1, x2, y2]
            landmarks = det['landmarks']
            score = det['score']

            if score < config.RETINA_CONFIDENCE_THRESHOLD:
                continue

            # Convert [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = facial_area
            box = (x1, y1, x2 - x1, y2 - y1)

            # Extract eye positions
            left_eye = (int(landmarks['left_eye'][0]), int(landmarks['left_eye'][1]))
            right_eye = (int(landmarks['right_eye'][0]), int(landmarks['right_eye'][1]))

            # Align
            aligned_image = align_face(image, left_eye, right_eye)

            # Crop for both models
            face_facenet = crop_and_resize(aligned_image, box, config.FACENET_INPUT_SIZE)
            face_arcface = crop_and_resize(aligned_image, box, config.ARCFACE_INPUT_SIZE)

            # Normalise landmark keys to match MTCNN format
            keypoints = {
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose': (int(landmarks['nose'][0]), int(landmarks['nose'][1])),
                'mouth_left': (int(landmarks['mouth_left'][0]), int(landmarks['mouth_left'][1])),
                'mouth_right': (int(landmarks['mouth_right'][0]), int(landmarks['mouth_right'][1]))
            }

            results.append({
                'box': box,
                'confidence': score,
                'landmarks': keypoints,
                'face_cropped': face_facenet,
                'face_cropped_arcface': face_arcface,
                'face_aligned': aligned_image
            })

        print(f"[RetinaFace] Detected {len(results)} face(s).")
        return results


# ══════════════════════════════════════════════
#  Factory Function — Pick detector by config
# ══════════════════════════════════════════════
def get_detector(model_name=None):
    """
    Returns the appropriate detector instance based on config or argument.

    Parameters:
        model_name (str, optional): "mtcnn" or "retinaface". Defaults to config setting.

    Returns:
        An instance of MTCNNDetector or RetinaFaceDetector.
    """
    name = model_name or config.DETECTION_MODEL

    if name.lower() == "mtcnn":
        return MTCNNDetector()
    elif name.lower() == "retinaface":
        return RetinaFaceDetector()
    else:
        raise ValueError(f"Unknown detector: {name}. Use 'mtcnn' or 'retinaface'.")


# ══════════════════════════════════════════════
#  Quick Test — Run this file directly to test
# ══════════════════════════════════════════════
if __name__ == "__main__":
    """
    Quick test: Run detection on a sample image or webcam frame.

    Usage:
        python detection.py                      # uses webcam
        python detection.py path/to/image.jpg    # uses a file
    """
    import sys

    # Load image
    if len(sys.argv) > 1:
        # Use provided image path
        img_path = sys.argv[1]
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Could not load image: {img_path}")
            sys.exit(1)
        print(f"[INFO] Loaded image: {img_path}")
    else:
        # Capture from webcam
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("[ERROR] Could not capture from webcam.")
            sys.exit(1)
        print("[INFO] Captured frame from webcam.")

    # Test MTCNN
    print("\n--- Testing MTCNN ---")
    mtcnn_det = MTCNNDetector()
    mtcnn_results = mtcnn_det.detect_faces(image)

    for i, face in enumerate(mtcnn_results):
        print(f"  Face {i+1}: confidence={face['confidence']:.4f}, box={face['box']}")
        print(f"    Landmarks: {face['landmarks']}")
        print(f"    Cropped shape (FaceNet): {face['face_cropped'].shape}")
        print(f"    Cropped shape (ArcFace): {face['face_cropped_arcface'].shape}")

        # Display the cropped face
        cv2.imshow(f"MTCNN Face {i+1}", face['face_cropped'])

        # Draw bounding box on original image
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks
        for name, point in face['landmarks'].items():
            cv2.circle(image, point, 3, (0, 0, 255), -1)

    # Show annotated image
    cv2.imshow("Detection Results", image)
    print("\n[INFO] Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Test RetinaFace
    print("\n--- Testing RetinaFace ---")
    try:
        retina_det = RetinaFaceDetector()
        retina_results = retina_det.detect_faces(image)
        for i, face in enumerate(retina_results):
            print(f"  Face {i+1}: confidence={face['confidence']:.4f}, box={face['box']}")
    except Exception as e:
        print(f"  [SKIP] RetinaFace test skipped: {e}")

    print("\n[DONE] Detection module test complete.")
