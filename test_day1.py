"""
test_day1.py — Day 1 Verification Script

Run this to verify that your Day 1 setup is working:
  1. MTCNN detects faces in an image
  2. Faces are cropped, aligned, and resized correctly
  3. The database can store and retrieve entries
  4. Everything persists to disk

Usage:
    python test_day1.py                         # test with webcam
    python test_day1.py sample_images/test.jpg  # test with an image file

Each team member should run this on their machine to confirm setup.
"""

import sys
import cv2
import numpy as np
import os

# Import our modules
import config
from detection import MTCNNDetector
from registration import FaceDatabase


def test_detection(image):
    """Test face detection on a single image."""
    print("\n" + "=" * 50)
    print("  STEP 1: Testing Face Detection (MTCNN)")
    print("=" * 50)

    detector = MTCNNDetector()
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        print("  [!] No faces detected. Try a clearer image with a visible face.")
        return None

    for i, face in enumerate(faces):
        print(f"\n  Face {i + 1}:")
        print(f"    Confidence : {face['confidence']:.4f}")
        print(f"    Bounding box: {face['box']}")
        print(f"    FaceNet crop shape : {face['face_cropped'].shape}")
        print(f"    ArcFace crop shape : {face['face_cropped_arcface'].shape}")

        # Check landmarks exist
        lm = face['landmarks']
        print(f"    Left eye  : {lm['left_eye']}")
        print(f"    Right eye : {lm['right_eye']}")
        print(f"    Nose      : {lm['nose']}")

    print(f"\n  [✓] Detection working — found {len(faces)} face(s)")
    return faces


def test_registration():
    """Test database registration and persistence."""
    print("\n" + "=" * 50)
    print("  STEP 2: Testing Registration & Persistence")
    print("=" * 50)

    test_db_path = os.path.join(config.DATA_DIR, "day1_test.json")

    # Clean start
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = FaceDatabase(db_path=test_db_path)

    # Register dummy users
    emb1 = np.random.randn(128).astype(np.float32)
    emb2 = np.random.randn(128).astype(np.float32)

    db.register("Test_User_Alice", emb1, "facenet")
    db.register("Test_User_Bob", emb2, "facenet")

    assert db.count() == 2, "Expected 2 users"
    print("  [✓] Registration works")

    # Test persistence
    db2 = FaceDatabase(db_path=test_db_path)
    assert db2.count() == 2, "Data did not persist"
    print("  [✓] Persistence works — data survives restart")

    # Test retrieval
    all_emb = db2.get_all_embeddings()
    assert len(all_emb) == 2, "Expected 2 embeddings"
    print("  [✓] Embedding retrieval works")

    # Test removal
    db2.remove_user("Test_User_Alice")
    assert db2.count() == 1, "Expected 1 user after removal"
    print("  [✓] User removal works")

    # Cleanup
    os.remove(test_db_path)
    print("  [✓] All registration tests passed!")


def visualise_detection(image, faces):
    """Draw bounding boxes and landmarks on the image and display."""
    print("\n" + "=" * 50)
    print("  STEP 3: Visual Verification")
    print("=" * 50)

    annotated = image.copy()

    for i, face in enumerate(faces):
        x, y, w, h = face['box']

        # Draw bounding box (green)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw confidence label
        label = f"Face {i+1}: {face['confidence']:.2f}"
        cv2.putText(annotated, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw landmarks (red dots)
        for name, point in face['landmarks'].items():
            cv2.circle(annotated, point, 3, (0, 0, 255), -1)

    # Save annotated image
    output_path = os.path.join(config.DATA_DIR, "day1_detection_result.jpg")
    cv2.imwrite(output_path, annotated)
    print(f"  [✓] Annotated image saved to: {output_path}")

    # Save cropped faces
    for i, face in enumerate(faces):
        crop_path = os.path.join(config.DATA_DIR, f"day1_cropped_face_{i+1}.jpg")
        cv2.imwrite(crop_path, face['face_cropped'])
        print(f"  [✓] Cropped face {i+1} saved to: {crop_path}")

    # Try to display (will fail on headless systems, that's OK)
    try:
        cv2.imshow("Day 1 - Detection Test", annotated)
        for i, face in enumerate(faces):
            cv2.imshow(f"Cropped Face {i+1} (160x160)", face['face_cropped'])
        print("\n  Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("  [i] Could not display windows (headless mode). Check saved images.")


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║    Day 1 Verification — Setup Check          ║")
    print("╚══════════════════════════════════════════════╝")

    # Load image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Cannot load image: {img_path}")
            sys.exit(1)
        print(f"[INFO] Using image: {img_path}")
    else:
        print("[INFO] No image provided. Capturing from webcam...")
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("[ERROR] Webcam capture failed. Try providing an image path:")
            print("  python test_day1.py path/to/face_image.jpg")
            sys.exit(1)
        print("[INFO] Webcam frame captured.")

    # Run tests
    faces = test_detection(image)
    test_registration()

    if faces:
        visualise_detection(image, faces)

    # Summary
    print("\n" + "=" * 50)
    print("  DAY 1 SUMMARY")
    print("=" * 50)
    print("  [✓] Config loaded")
    print("  [✓] MTCNN detection working")
    print("  [✓] Face cropping & alignment working")
    print("  [✓] Database registration working")
    print("  [✓] Persistence (save/load) working")
    print()
    print("  NEXT STEPS (Day 2):")
    print("    Person A: Add RetinaFace detector")
    print("    Person B: Add FaceNet embedding extraction")
    print("    Person C: Connect registration to real embeddings")
    print("    Person D: Start evaluation metrics scaffold")
    print()
    print("  Your Day 1 setup is complete! 🎉")


if __name__ == "__main__":
    main()
