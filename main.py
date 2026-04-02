"""
main.py — Smart Office Face Recognition System

Entry point for the application. Provides a CLI menu for:
  1. Register a new person
  2. Recognise a face
  3. List registered users
  4. Remove a user
  5. Quit

This file ties together all modules:
  - detection.py   (Person A)
  - recognition.py (Person B)
  - registration.py (Person C)
  - evaluation.py  (Person D)

Usage:
    python main.py
"""

import cv2
import sys
import config
from detection import get_detector
from registration import FaceDatabase
# from recognition import get_recogniser  # Uncomment on Day 2 when Person B is ready


def capture_image():
    """
    Capture a single frame from the webcam or load from file.

    Returns:
        numpy.ndarray: The captured image in BGR format, or None on failure.
    """
    print("\nCapture options:")
    print("  1. Use webcam")
    print("  2. Load from file")
    choice = input("Choose (1/2): ").strip()

    if choice == "2":
        path = input("Enter image path: ").strip()
        image = cv2.imread(path)
        if image is None:
            print(f"[ERROR] Cannot load: {path}")
            return None
        return image
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        print("[INFO] Webcam opening... Press 'c' to capture, 'q' to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam not available.")
                break
            cv2.imshow("Capture - Press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                return frame
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None


def register_flow(detector, db):
    """Register a new person."""
    print("\n--- Register New Person ---")
    name = input("Enter person's name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return

    image = capture_image()
    if image is None:
        return

    # Detect face
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        print("[ERROR] No face detected. Try again with a clearer image.")
        return

    if len(faces) > 1:
        print(f"[WARNING] {len(faces)} faces detected. Using the largest one.")
        # Pick the face with the largest bounding box
        faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)

    face = faces[0]
    print(f"[INFO] Face detected with confidence: {face['confidence']:.4f}")

    # TODO: Extract embedding (Person B — Day 2)
    # recogniser = get_recogniser()
    # embedding = recogniser.extract_embedding(face['face_cropped'])
    # For now, use a dummy embedding to test the flow
    import numpy as np
    dummy_embedding = np.random.randn(128).astype(np.float32)

    # Register
    status = db.register(name, dummy_embedding, config.RECOGNITION_MODEL, face['face_cropped'])
    print(status)
    print(f"[INFO] '{name}' registered successfully!")


def recognise_flow(detector, db):
    """Recognise a face against the database."""
    print("\n--- Recognise Face ---")

    if db.is_empty():
        print("[INFO] No registered users yet. Register someone first.")
        return

    image = capture_image()
    if image is None:
        return

    faces = detector.detect_faces(image)
    if len(faces) == 0:
        print("[ERROR] No face detected.")
        return

    face = faces[0]
    print(f"[INFO] Face detected with confidence: {face['confidence']:.4f}")

    # TODO: Extract embedding and match (Person B — Day 3)
    # recogniser = get_recogniser()
    # embedding = recogniser.extract_embedding(face['face_cropped'])
    # all_known = db.get_all_embeddings()
    # best_match, best_score = match_embedding(embedding, all_known)

    print("[INFO] Recognition matching — coming on Day 3 (Person B).")
    print("[INFO] For now, detection and registration are verified working.")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Smart Office Face Recognition System              ║")
    print("║    Employee Access Control                           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Detection model : {config.DETECTION_MODEL}")
    print(f"  Recognition model: {config.RECOGNITION_MODEL}")
    print()

    # Initialise modules
    detector = get_detector()
    db = FaceDatabase()

    while True:
        print("\n──── Main Menu ────")
        print("  1. Register new person")
        print("  2. Recognise a face")
        print("  3. List registered users")
        print("  4. Remove a user")
        print("  5. Quit")
        choice = input("\nChoose an option (1-5): ").strip()

        if choice == "1":
            register_flow(detector, db)
        elif choice == "2":
            recognise_flow(detector, db)
        elif choice == "3":
            db.list_users()
        elif choice == "4":
            name = input("Enter name to remove: ").strip()
            db.remove_user(name)
        elif choice == "5":
            print("\nGoodbye! 👋")
            break
        else:
            print("[ERROR] Invalid option. Choose 1-5.")


if __name__ == "__main__":
    main()
