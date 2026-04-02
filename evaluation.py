"""
evaluation.py — Evaluation & Metrics Module (Person D's responsibility)

This module will handle:
  1. Computing accuracy metrics for face recognition
  2. Generating confusion matrices
  3. Threshold analysis (finding the optimal similarity threshold)
  4. Comparing performance across model combinations

TODO (Person D — Day 3-5):
  - Implement accuracy calculation
  - Implement confusion matrix generation
  - Add model comparison reporting
  - Generate plots/charts for the report
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import config


def compute_accuracy(true_labels, predicted_labels):
    """
    Compute recognition accuracy.

    Parameters:
        true_labels (list): Ground truth names.
        predicted_labels (list): Predicted names from the system.

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    return accuracy_score(true_labels, predicted_labels)


def generate_confusion_matrix(true_labels, predicted_labels, label_names=None):
    """
    Generate a confusion matrix for recognition results.

    Parameters:
        true_labels (list): Ground truth names.
        predicted_labels (list): Predicted names.
        label_names (list, optional): Ordered list of person names.

    Returns:
        numpy.ndarray: The confusion matrix.
    """
    return confusion_matrix(true_labels, predicted_labels, labels=label_names)


def model_comparison_report(results_dict):
    """
    Compare performance across different model combinations.

    Parameters:
        results_dict (dict): {
            "MTCNN + FaceNet": {"accuracy": 0.95, "true": [...], "pred": [...]},
            "MTCNN + ArcFace": {"accuracy": 0.97, "true": [...], "pred": [...]},
            ...
        }

    TODO: Implement reporting and visualisation.
    """
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          Model Comparison Report                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    for combo_name, data in results_dict.items():
        acc = data.get('accuracy', 0)
        print(f"║  {combo_name:<30} | Accuracy: {acc:.2%}    ║")
    print("╚══════════════════════════════════════════════════════╝")
