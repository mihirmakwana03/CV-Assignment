"""
evaluation.py - Evaluation & Metrics
Computes accuracy, confusion matrices, and model comparison reports.

Person D is responsible for this module.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def compute_accuracy(true_labels, predicted_labels):
    """Simple accuracy: what fraction of predictions were correct."""
    return accuracy_score(true_labels, predicted_labels)


def generate_confusion_matrix(true_labels, predicted_labels, label_names=None):
    """Build a confusion matrix for recognition results."""
    return confusion_matrix(true_labels, predicted_labels, labels=label_names)


def get_classification_report(true_labels, predicted_labels):
    """Detailed per-class precision, recall, F1."""
    return classification_report(true_labels, predicted_labels)


def compare_models(results_dict):
    """
    Print a comparison of model combinations.
    
    results_dict format:
        {"MTCNN + FaceNet": {"accuracy": 0.95, ...}, ...}
    """
    print("\nModel comparison:")
    print(f"  {'Combination':<30} {'Accuracy':>10}")
    print("  " + "-" * 42)
    for combo, data in results_dict.items():
        print(f"  {combo:<30} {data['accuracy']:>10.2%}")
    print()
