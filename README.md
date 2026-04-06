# Smart Office Face Recognition System

Face detection and recognition system for employee access control.
Detects faces at an office door, identifies known employees, and registers new visitors.

## Setup

```bash
pip install torch torchvision
pip install facenet-pytorch --no-deps
pip install opencv-python scikit-learn Pillow
pip install retina-face --no-deps    # optional, for RetinaFace comparison
pip install deepface --no-deps       # optional, for ArcFace via deepface
```

## Quick test

```bash
python test_setup.py sample_images/your_photo.jpg
```

## Run the system

```bash
python main.py
```

## Run detector comparison (Person A)

```bash
python compare_detectors.py
```

Put test images in `sample_images/` first.

## Project structure

```
config.py             - All settings in one place
detection.py          - Person A: MTCNN + RetinaFace face detection
recognition.py        - Person B: FaceNet + ArcFace embeddings (TODO)
registration.py       - Person C: JSON database, registration flow
evaluation.py         - Person D: Accuracy metrics, confusion matrix (TODO)
main.py               - CLI entry point, ties everything together
compare_detectors.py  - Person A: MTCNN vs RetinaFace comparison
test_setup.py         - Quick verification script
```

## How detection output works

Both detectors return the same format so recognition code works with either:

```python
from detection import get_detector

detector = get_detector("mtcnn")  # or "retinaface"
faces = detector.detect(image)

for face in faces:
    face['box']         # (x, y, w, h) bounding box
    face['confidence']  # 0.0 to 1.0
    face['landmarks']   # dict with left_eye, right_eye, nose, mouth_left, mouth_right
    face['face_160']    # 160x160 BGR crop (for FaceNet)
    face['face_112']    # 112x112 BGR crop (for ArcFace)
    face['detector']    # "mtcnn" or "retinaface"
    face['time_ms']     # detection time in milliseconds
```

## Team

- Person A: Face detection (detection.py, compare_detectors.py)
- Person B: Face recognition (recognition.py)
- Person C: Registration & integration (registration.py, main.py)
- Person D: Evaluation & GUI (evaluation.py)
