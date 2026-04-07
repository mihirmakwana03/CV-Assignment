"""
evaluation.py - Evaluation & Metrics
Computes accuracy, confusion matrices, and model comparison reports.

Person D 
"""

import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from detection import get_detector
from recognition import get_recogniser
from registration import FaceDatabase


def compute_accuracy(true_labels, predicted_labels):
    """Simple accuracy: fraction of predictions that were correct."""
    return accuracy_score(true_labels, predicted_labels)


def generate_confusion_matrix(true_labels, predicted_labels, label_names=None):
    """Build a confusion matrix for recognition results."""
    return confusion_matrix(true_labels, predicted_labels, labels=label_names)


def get_classification_report(true_labels, predicted_labels):
    """Detailed per-class precision, recall, F1."""
    return classification_report(true_labels, predicted_labels)


def compare_models(results_dict):
    """
    Print comparison of model combinations.
    Example:
        {"MTCNN + FaceNet": {"accuracy": 0.95}}
    """
    print("\nModel comparison:")
    print(f"{'Combination':<30} {'Accuracy':>10}")
    print("-" * 42)

    for combo, data in results_dict.items():
        print(f"{combo:<30} {data['accuracy']:>10.2%}")

    print()


def run_evaluation(dataset_path="sample_images"):
    """
    Run evaluation on all images in dataset_path.
    Image naming format: personname_x.jpg
    Example: shreyas_1.jpg
    """

    print("\nRunning evaluation...")

    detector = get_detector()
    recogniser = get_recogniser()
    db = FaceDatabase()

    true_labels = []
    predicted_labels = []

    images = os.listdir(dataset_path)

    for img_name in images:

        path = os.path.join(dataset_path, img_name)

        image = cv2.imread(path)

        if image is None:
            continue

        faces = detector.detect(image)

        for face in faces:

            embedding = recogniser.extract_embedding(face["face_160"])

            best_name = "Unknown"
            best_score = 0

            for name, db_emb in db.get_all_embeddings():

                if len(embedding) != len(db_emb):
                    continue

                score = recogniser.compare(embedding, db_emb)

                if score > best_score:
                    best_score = score
                    best_name = name

            # ground truth from filename
            true_name = img_name.split("_")[0]

            true_labels.append(true_name)
            predicted_labels.append(best_name)

    accuracy = compute_accuracy(true_labels, predicted_labels)

    print("\nEvaluation Results")
    print("------------------")

    print("Accuracy:", accuracy)

    labels = list(set(true_labels))

    cm = generate_confusion_matrix(true_labels, predicted_labels, labels)

    print("\nConfusion Matrix:")
    print(cm)

    report = get_classification_report(true_labels, predicted_labels)

    print("\nClassification Report:")
    print(report)

    return accuracy


if __name__ == "__main__":
    run_evaluation()