"""
live_detection.py - Live Webcam Face Detection & Recognition
Continuous face detection from webcam with real-time recognition.

Controls:
    r - Register the currently detected face
    q - Quit
    p - Toggle preprocessing on/off

Usage:
    python live_detection.py
"""

import cv2
import time
import numpy as np
import config
from detection import get_detector
from registration import FaceDatabase
from recognition import get_recogniser


def draw_face(frame, face, label="Unknown", colour=(0, 200, 0)):
    """Draw bounding box, landmarks and name on the frame."""
    x, y, w, h = [int(v) for v in face['box']]

    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    text = f"{label} ({face['confidence']:.0%})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), colour, -1)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    for pt in face['landmarks'].values():
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)


def draw_status(frame, fps, num_faces, preprocessing):
    """Info bar at top and controls at bottom."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 35), (40, 40, 40), -1)
    status = f"FPS: {fps:.1f} | Faces: {num_faces} | Preprocessing: {'ON' if preprocessing else 'OFF'}"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.rectangle(frame, (0, h - 30), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, "R=Register | Q=Quit | P=Toggle preprocessing",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def match_face(face, recogniser, db):
    """
    Extract embedding from detected face and compare against database.
    Returns (name, similarity_score).
    """
    if db.is_empty():
        return "Unknown", 0.0

    # extract embedding using Person B's recogniser
    embedding = recogniser.extract_embedding(face['face_160'])

    known = db.get_all_embeddings()
    best_name = "Unknown"
    best_score = 0.0

    for name, known_emb in known:
        score = recogniser.compare(embedding, known_emb)
        if score > config.FACENET_THRESHOLD and score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score


def register_face(face, recogniser, db):
    """Register a face with real embedding extraction."""
    name = input("\n  Enter name to register: ").strip()
    if not name:
        print("  Cancelled")
        return

    embedding = recogniser.extract_embedding(face['face_160'])
    db.register(name, embedding, config.RECOGNITION_MODEL, face['face_160'])
    print(f"  Registered: {name}")


def main():
    print("\nLive Face Detection & Recognition")
    print("Starting...\n")

    db = FaceDatabase()
    detector = get_detector()
    recogniser = get_recogniser()
    use_preprocessing = True

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # detect every 0.3s instead of every frame for performance
    last_detect = 0
    detect_interval = 0.3
    cached_faces = []
    cached_labels = []

    fps = 0
    frame_count = 0
    fps_timer = time.time()

    print("Controls: R=Register, Q=Quit, P=Toggle preprocessing\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost webcam")
            break

        now = time.time()

        # run detection + recognition at intervals
        if now - last_detect >= detect_interval:
            cached_faces = detector.detect(frame, use_preprocessing=use_preprocessing)
            cached_labels = []
            for face in cached_faces:
                name, score = match_face(face, recogniser, db)
                cached_labels.append((name, score))
            last_detect = now

        # draw on every frame for smooth video
        display = frame.copy()
        for i, face in enumerate(cached_faces):
            if i < len(cached_labels):
                name, score = cached_labels[i]
            else:
                name, score = "Unknown", 0.0

            if name != "Unknown":
                label = f"{name} ({score:.0%})"
                colour = (0, 200, 0)    # green = known
            else:
                label = "Unknown"
                colour = (0, 140, 255)  # orange = unknown

            draw_face(display, face, label, colour)

        # FPS counter
        frame_count += 1
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        draw_status(display, fps, len(cached_faces), use_preprocessing)
        cv2.imshow("Smart Office - Face Recognition", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if cached_faces:
                register_face(cached_faces[0], recogniser, db)
            else:
                print("  No face visible to register")
        elif key == ord('p'):
            use_preprocessing = not use_preprocessing
            print(f"  Preprocessing: {'ON' if use_preprocessing else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
