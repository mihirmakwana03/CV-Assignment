"""
test_setup.py - Quick verification that everything works.
Each team member should run this after cloning the project.

Usage:
    python test_setup.py                        # webcam
    python test_setup.py sample_images/pic.jpg  # from file
"""

import sys
import os
import cv2
import numpy as np
import config
from detection import MTCNNDetector
from registration import FaceDatabase


def main():
    print("Setup verification\n")

    # load an image
    if len(sys.argv) > 1:
        image = cv2.imread(sys.argv[1])
        if image is None:
            print(f"Cannot load: {sys.argv[1]}")
            return
        print(f"Loaded: {sys.argv[1]}")
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print("Webcam failed. Pass an image path instead:")
            print("  python test_setup.py sample_images/your_photo.jpg")
            return
        print("Captured from webcam")

    # test detection
    print("\nTesting MTCNN detection...")
    detector = MTCNNDetector()
    faces = detector.detect(image)

    if not faces:
        print("  No faces found. Use a clearer image with a visible face.")
        return

    for i, f in enumerate(faces):
        print(f"  Face {i+1}: confidence={f['confidence']:.4f}, "
              f"box={f['box']}, time={f['time_ms']:.0f}ms")
        print(f"    face_160 shape: {f['face_160'].shape}")
        print(f"    face_112 shape: {f['face_112'].shape}")

    # save cropped face for inspection
    out_path = os.path.join(config.DATA_DIR, "test_crop.jpg")
    cv2.imwrite(out_path, faces[0]['face_160'])
    print(f"\n  Cropped face saved: {out_path}")

    # test database
    print("\nTesting database...")
    test_db = os.path.join(config.DATA_DIR, "_test_db.json")
    db = FaceDatabase(db_path=test_db)

    db.register("Test User", np.random.randn(128).astype(np.float32), "facenet")
    assert db.count() == 1

    db2 = FaceDatabase(db_path=test_db)
    assert db2.count() == 1
    print("  Persistence works")

    db2.remove_user("Test User")
    assert db2.count() == 0
    os.remove(test_db)
    print("  Register/remove works")

    print("\nAll checks passed. Project is ready.")


if __name__ == "__main__":
    main()
