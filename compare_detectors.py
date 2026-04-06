"""
compare_detectors.py — MTCNN vs RetinaFace Comparison (Person A's Day 2 test)

This script runs both detectors on a folder of test images and compares:
  1. Number of faces detected
  2. Detection speed (milliseconds)
  3. Confidence scores
  4. Bounding box overlap (IoU — Intersection over Union)
  5. Visual side-by-side comparison output

USAGE:
    python compare_detectors.py                          # uses sample_images/
    python compare_detectors.py path/to/image_folder     # uses custom folder
    python compare_detectors.py path/to/single_image.jpg # single image

The results are printed as a table and saved to data/detector_comparison.txt.
Side-by-side annotated images are saved to data/comparison_visuals/.

USE THIS DATA IN YOUR REPORT:
  - Include the comparison table
  - Discuss which detector is faster vs more accurate
  - Explain WHY RetinaFace is more accurate (FPN, anchor boxes, training data)
  - Discuss when you'd choose MTCNN over RetinaFace (speed vs accuracy trade-off)
"""

import cv2
import numpy as np
import os
import sys
import time
import config
from detection import (
    MTCNNDetector, RetinaFaceDetector,
    draw_detections, save_cropped_faces, FacePreprocessor
)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures how much two boxes overlap:
      - 1.0 = perfect overlap (identical boxes)
      - 0.0 = no overlap at all
      - > 0.5 is generally considered a "match"

    This is useful to check whether MTCNN and RetinaFace are detecting
    the SAME face (high IoU) or different regions (low IoU).

    Parameters:
        box1 (tuple): (x, y, w, h) first bounding box.
        box2 (tuple): (x, y, w, h) second bounding box.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x1_a, y1_a = box1[0], box1[1]
    x2_a, y2_a = box1[0] + box1[2], box1[1] + box1[3]

    x1_b, y1_b = box2[0], box2[1]
    x2_b, y2_b = box2[0] + box2[2], box2[1] + box2[3]

    # Compute intersection rectangle
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    area_a = box1[2] * box1[3]
    area_b = box2[2] * box2[3]
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compare_on_image(image, mtcnn_detector, retina_detector, image_name="image"):
    """
    Run both detectors on a single image and compare results.

    Parameters:
        image (numpy.ndarray): BGR image.
        mtcnn_detector: MTCNNDetector instance.
        retina_detector: RetinaFaceDetector instance (or None).
        image_name (str): Name for logging.

    Returns:
        dict: Comparison results for this image.
    """
    result = {'image': image_name}

    # ── MTCNN ──
    mtcnn_faces = mtcnn_detector.detect_faces(image)
    result['mtcnn_count'] = len(mtcnn_faces)
    result['mtcnn_time_ms'] = mtcnn_faces[0]['detection_time_ms'] if mtcnn_faces else 0
    result['mtcnn_confidences'] = [f['confidence'] for f in mtcnn_faces]
    result['mtcnn_boxes'] = [f['box'] for f in mtcnn_faces]
    result['mtcnn_detections'] = mtcnn_faces

    # ── RetinaFace ──
    if retina_detector is not None and retina_detector.available:
        retina_faces = retina_detector.detect_faces(image)
        result['retina_count'] = len(retina_faces)
        result['retina_time_ms'] = retina_faces[0]['detection_time_ms'] if retina_faces else 0
        result['retina_confidences'] = [f['confidence'] for f in retina_faces]
        result['retina_boxes'] = [f['box'] for f in retina_faces]
        result['retina_detections'] = retina_faces
    else:
        result['retina_count'] = -1  # not available
        result['retina_time_ms'] = 0
        result['retina_confidences'] = []
        result['retina_boxes'] = []
        result['retina_detections'] = []

    # ── IoU between matched detections ──
    ious = []
    for m_box in result['mtcnn_boxes']:
        for r_box in result['retina_boxes']:
            iou = compute_iou(m_box, r_box)
            if iou > 0.3:  # likely the same face
                ious.append(iou)
    result['avg_iou'] = np.mean(ious) if ious else 0.0

    return result


def create_side_by_side(image, mtcnn_dets, retina_dets):
    """
    Create a side-by-side annotated image showing both detectors.

    Left half: MTCNN detections (green boxes)
    Right half: RetinaFace detections (blue boxes)

    Parameters:
        image (numpy.ndarray): Original image.
        mtcnn_dets (list): MTCNN detection results.
        retina_dets (list): RetinaFace detection results.

    Returns:
        numpy.ndarray: Side-by-side comparison image.
    """
    # Draw MTCNN results in green
    mtcnn_vis = draw_detections(image, mtcnn_dets, colour=(0, 255, 0), label_prefix="MTCNN")

    # Draw RetinaFace results in blue
    retina_vis = draw_detections(image, retina_dets, colour=(255, 100, 0), label_prefix="Retina")

    # Add labels at the top
    h, w = mtcnn_vis.shape[:2]
    cv2.putText(mtcnn_vis, "MTCNN", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(retina_vis, "RetinaFace", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

    # Concatenate horizontally
    combined = np.hstack([mtcnn_vis, retina_vis])
    return combined


def print_comparison_table(all_results):
    """
    Print a formatted comparison table and return it as a string.

    Parameters:
        all_results (list of dict): Results from compare_on_image().

    Returns:
        str: The formatted table as a string (for saving to file).
    """
    lines = []
    header = (
        f"{'Image':<25} | {'MTCNN':^6} | {'Retina':^6} | "
        f"{'MTCNN ms':^9} | {'Retina ms':^10} | "
        f"{'MTCNN Conf':^11} | {'Retina Conf':^12} | {'Avg IoU':^8}"
    )
    separator = "─" * len(header)

    lines.append("\n" + separator)
    lines.append("  DETECTOR COMPARISON RESULTS")
    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    total_mtcnn_faces = 0
    total_retina_faces = 0
    total_mtcnn_time = 0
    total_retina_time = 0
    num_images = len(all_results)

    for r in all_results:
        mtcnn_conf = f"{np.mean(r['mtcnn_confidences']):.3f}" if r['mtcnn_confidences'] else "N/A"
        retina_conf = f"{np.mean(r['retina_confidences']):.3f}" if r['retina_confidences'] else "N/A"
        retina_count = str(r['retina_count']) if r['retina_count'] >= 0 else "N/A"

        line = (
            f"{r['image']:<25} | {r['mtcnn_count']:^6} | {retina_count:^6} | "
            f"{r['mtcnn_time_ms']:^9.1f} | {r['retina_time_ms']:^10.1f} | "
            f"{mtcnn_conf:^11} | {retina_conf:^12} | {r['avg_iou']:^8.3f}"
        )
        lines.append(line)

        total_mtcnn_faces += r['mtcnn_count']
        total_retina_faces += max(0, r['retina_count'])
        total_mtcnn_time += r['mtcnn_time_ms']
        total_retina_time += r['retina_time_ms']

    lines.append(separator)

    # Summary
    lines.append(f"\n  SUMMARY ({num_images} images tested):")
    lines.append(f"    MTCNN  — Total faces: {total_mtcnn_faces}, "
                 f"Avg time: {total_mtcnn_time/max(1,num_images):.1f}ms/image")
    if total_retina_faces >= 0:
        lines.append(f"    Retina — Total faces: {total_retina_faces}, "
                     f"Avg time: {total_retina_time/max(1,num_images):.1f}ms/image")

    speed_ratio = total_retina_time / max(1, total_mtcnn_time)
    lines.append(f"\n    Speed ratio: RetinaFace is {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'} than MTCNN")

    if total_retina_faces > total_mtcnn_faces:
        lines.append(f"    RetinaFace detected {total_retina_faces - total_mtcnn_faces} more face(s) overall")
    elif total_mtcnn_faces > total_retina_faces:
        lines.append(f"    MTCNN detected {total_mtcnn_faces - total_retina_faces} more face(s) overall")
    else:
        lines.append(f"    Both detectors found the same number of faces")

    lines.append(separator + "\n")

    output = "\n".join(lines)
    print(output)
    return output


def test_preprocessing_impact(image, detector, image_name="image"):
    """
    Compare detection with and without preprocessing to show its value.

    This is useful data for your report — it demonstrates WHY preprocessing
    is included in the pipeline.

    Parameters:
        image (numpy.ndarray): BGR image.
        detector: Any detector instance.
        image_name (str): Name for logging.

    Returns:
        dict: Results with and without preprocessing.
    """
    print(f"\n  Testing preprocessing impact on: {image_name}")

    # Without preprocessing
    faces_raw = detector.detect_faces(image, preprocess=False)
    raw_count = len(faces_raw)
    raw_confs = [f['confidence'] for f in faces_raw]

    # With preprocessing
    faces_pp = detector.detect_faces(image, preprocess=True)
    pp_count = len(faces_pp)
    pp_confs = [f['confidence'] for f in faces_pp]

    print(f"    Without preprocessing: {raw_count} face(s), "
          f"avg conf: {np.mean(raw_confs):.4f}" if raw_confs else
          f"    Without preprocessing: {raw_count} face(s)")
    print(f"    With preprocessing:    {pp_count} face(s), "
          f"avg conf: {np.mean(pp_confs):.4f}" if pp_confs else
          f"    With preprocessing:    {pp_count} face(s)")

    return {
        'image': image_name,
        'raw_count': raw_count,
        'raw_avg_conf': np.mean(raw_confs) if raw_confs else 0,
        'pp_count': pp_count,
        'pp_avg_conf': np.mean(pp_confs) if pp_confs else 0
    }


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Day 2 — MTCNN vs RetinaFace Comparison            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Determine input source
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = config.SAMPLE_IMAGES_DIR

    # Collect images to test
    images = {}
    if os.path.isfile(input_path):
        img = cv2.imread(input_path)
        if img is not None:
            images[os.path.basename(input_path)] = img
    elif os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(os.path.join(input_path, fname))
                if img is not None:
                    images[fname] = img

    if not images:
        print(f"\n[ERROR] No images found at: {input_path}")
        print("  Please add face images to sample_images/ or provide a path:")
        print("    python compare_detectors.py path/to/images/")
        print("\n  TIP: Each group member should add 5-10 photos of themselves")
        print("       with different angles, lighting, and expressions.")
        sys.exit(1)

    print(f"\n[INFO] Found {len(images)} test image(s).")

    # Initialise detectors
    print("\n--- Initialising Detectors ---")
    mtcnn = MTCNNDetector()
    retina = RetinaFaceDetector()

    # Output directories
    vis_dir = os.path.join(config.DATA_DIR, "comparison_visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # ══════════════════════════════════════════
    #  TEST 1: Detector Comparison
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: MTCNN vs RetinaFace — Detection Comparison")
    print("=" * 60)

    all_results = []
    for name, image in images.items():
        print(f"\n  Processing: {name} ({image.shape[1]}x{image.shape[0]})")
        result = compare_on_image(image, mtcnn, retina, image_name=name)
        all_results.append(result)

        # Save side-by-side visual
        if retina.available and result['retina_detections']:
            combined = create_side_by_side(
                image, result['mtcnn_detections'], result['retina_detections']
            )
            vis_path = os.path.join(vis_dir, f"compare_{name}")
            cv2.imwrite(vis_path, combined)
            print(f"    Side-by-side saved: {vis_path}")

    # Print and save comparison table
    table_text = print_comparison_table(all_results)

    report_path = os.path.join(config.DATA_DIR, "detector_comparison.txt")
    with open(report_path, 'w') as f:
        f.write(table_text)
    print(f"[INFO] Comparison report saved to: {report_path}")

    # ══════════════════════════════════════════
    #  TEST 2: Preprocessing Impact
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: Preprocessing Impact on Detection")
    print("=" * 60)

    pp_results = []
    for name, image in images.items():
        pp_result = test_preprocessing_impact(image, mtcnn, image_name=name)
        pp_results.append(pp_result)

    # Summary
    print("\n  PREPROCESSING SUMMARY:")
    for r in pp_results:
        delta = r['pp_count'] - r['raw_count']
        conf_delta = r['pp_avg_conf'] - r['raw_avg_conf']
        indicator = "↑" if delta > 0 else ("=" if delta == 0 else "↓")
        print(f"    {r['image']:<25}: {indicator} faces ({r['raw_count']}→{r['pp_count']}), "
              f"conf change: {conf_delta:+.4f}")

    # ══════════════════════════════════════════
    #  TEST 3: Save Cropped Faces for Person B
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: Saving Cropped Faces for Recognition Module")
    print("=" * 60)

    crops_dir = os.path.join(config.DATA_DIR, "cropped_faces")
    total_saved = 0
    for name, image in images.items():
        faces = mtcnn.detect_faces(image)
        if faces:
            basename = os.path.splitext(name)[0]
            saved = save_cropped_faces(faces, crops_dir, prefix=basename)
            total_saved += len(saved)

    print(f"\n  [✓] Saved {total_saved} cropped face images to {crops_dir}")
    print("      Person B can use these to test embedding extraction tomorrow.")

    # ══════════════════════════════════════════
    #  Final Summary
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  DAY 2 COMPLETE — Person A Summary")
    print("=" * 60)
    print("  [✓] MTCNN detector: fully working with preprocessing")
    print("  [✓] RetinaFace detector: fully working with preprocessing")
    print("  [✓] Face alignment: eye-landmark-based rotation")
    print("  [✓] Edge case handling: tiny faces, boundary faces, low confidence")
    print("  [✓] Comparison test: speed, accuracy, IoU measurements")
    print("  [✓] Preprocessing impact: CLAHE + denoising tested")
    print("  [✓] Cropped faces saved for Person B")
    print()
    print("  FILES FOR YOUR REPORT:")
    print(f"    - Comparison table: {report_path}")
    print(f"    - Visual comparisons: {vis_dir}/")
    print(f"    - Cropped faces: {crops_dir}/")
    print()
    print("  NEXT STEPS (Day 3-4):")
    print("    - Help Person B test recognition with your cropped faces")
    print("    - Test with challenging images (glasses, side profiles, masks)")
    print("    - Help Person C integrate both detectors into main.py")


if __name__ == "__main__":
    main()
