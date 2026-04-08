"""
main.py - Smart Office Face Recognition System
CLI interface for registering and recognising faces.
"""

import cv2
import numpy as np
import config
from detection import get_detector
from registration import FaceDatabase
from recognition import get_recogniser


def capture_image():
    """Grab an image from webcam or file."""
    print("\n  1. Webcam")
    print("  2. Load from file")
    choice = input("  Choose (1/2): ").strip()

    if choice == "2":
        path = input("  Image path: ").strip()
        img = cv2.imread(path)
        if img is None:
            print(f"  Could not load: {path}")
        return img

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("  Press 'c' to capture, 'q' to cancel")
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            print("  Webcam not available")
            break
        cv2.imshow("Capture", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            frame = f
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def register_person(detector, recogniser, db):
    """Register a new person into the database."""
    name = input("\n  Name: ").strip()
    if not name:
        print("  Name cannot be empty")
        return

    image = capture_image()
    if image is None:
        return

    faces = detector.detect(image)
    if not faces:
        print("  No face detected. Try again with a clearer image.")
        return

    if len(faces) > 1:
        faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)
        print(f"  Found {len(faces)} faces, using the largest one")

    face = faces[0]
    print(f"  Face detected (confidence: {face['confidence']:.3f})")

    # extract real embedding using Person B's recogniser
    embedding = recogniser.extract_embedding(face['face_160'])
    db.register(name, embedding, config.RECOGNITION_MODEL, face['face_160'])
    print(f"  Registered: {name}")


def recognise_person(detector, recogniser, db):
    """Try to identify a face against the database."""
    if db.is_empty():
        print("\n  No registered users yet. Register someone first.")
        return

    image = capture_image()
    if image is None:
        return

    faces = detector.detect(image)
    if not faces:
        print("  No face detected.")
        return

    face = faces[0]
    print(f"  Face detected (confidence: {face['confidence']:.3f})")

    # extract embedding and match against database
    embedding = recogniser.extract_embedding(face['face_160'])
    known = db.get_all_embeddings()

    best_name = "Unknown"
    best_score = 0.0

    for name, known_emb in known:
        # skip if dimensions don't match (mixed model entries)
        if np.asarray(embedding).shape[0] != np.asarray(known_emb).shape[0]:
            continue
        score = recogniser.compare(embedding, known_emb)
        if score > config.FACENET_THRESHOLD and score > best_score:
            best_name = name
            best_score = score

    if best_name != "Unknown":
        print(f"  Welcome back, {best_name}! (similarity: {best_score:.3f})")
    else:
        print("  Unknown person. Use option 1 to register them.")


def main():
    print("\nSmart Office Face Recognition System")
    print(f"  Detection: {config.DETECTION_MODEL}")
    print(f"  Recognition: {config.RECOGNITION_MODEL}\n")

    detector = get_detector()
    recogniser = get_recogniser()
    db = FaceDatabase()

    while True:
        print("\n  1. Register new person")
        print("  2. Recognise a face")
        print("  3. Live webcam mode")
        print("  4. List registered users")
        print("  5. Remove a user")
        print("  6. Run evaluation")
        print("  7. Quit")
        choice = input("\n  > ").strip()

        if choice == "1":
            register_person(detector, recogniser, db)
        elif choice == "2":
            recognise_person(detector, recogniser, db)
        elif choice == "3":
            from live_detection import main as live_main
            live_main()
        elif choice == "4":
            db.list_users()
        elif choice == "5":
            name = input("  Name to remove: ").strip()
            db.remove_user(name)
        elif choice == "6":
            from evaluation import run_evaluation
            run_evaluation()
        elif choice == "7":
            print("  Bye!")
            break
        else:
            print("  Pick 1-7")


if __name__ == "__main__":
    main()