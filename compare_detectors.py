"""
compare_detectors.py - Run MTCNN and RetinaFace on test images and compare.
Outputs a comparison table, saves annotated images and cropped faces.

Usage:
    python compare_detectors.py                      # uses sample_images/
    python compare_detectors.py path/to/folder       # custom folder
    python compare_detectors.py path/to/image.jpg    # single image
"""

import cv2
import numpy as np
import os
import sys
import config
from detection import MTCNNDetector, RetinaFaceDetector, Preprocessor


def load_images(path):
    """Load images from a file path or directory."""
    images = {}
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            images[os.path.basename(path)] = img
    elif os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(os.path.join(path, f))
                if img is not None:
                    images[f] = img
    return images


def compute_iou(box_a, box_b):
    """Intersection over Union between two (x, y, w, h) boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(image, detections, colour, label):
    """Draw bounding boxes and landmarks on a copy of the image."""
    out = image.copy()
    for d in detections:
        x, y, w, h = [int(v) for v in d['box']]
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(out, f"{label} {d['confidence']:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        for pt in d['landmarks'].values():
            cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    return out


def save_crops(detections, folder, prefix):
    """Save cropped faces (both 160x160 and 112x112) to disk."""
    os.makedirs(folder, exist_ok=True)
    for i, d in enumerate(detections):
        cv2.imwrite(os.path.join(folder, f"{prefix}_{i+1}_160.jpg"), d['face_160'])
        cv2.imwrite(os.path.join(folder, f"{prefix}_{i+1}_112.jpg"), d['face_112'])


def run_comparison(images, mtcnn, retina):
    """Run both detectors on all images and collect results."""
    rows = []
    for name, img in images.items():
        print(f"  {name} ({img.shape[1]}x{img.shape[0]})")

        m = mtcnn.detect(img)
        r = retina.detect(img) if retina.available else []

        # find average IoU for matched faces
        ious = []
        for mb in [f['box'] for f in m]:
            for rb in [f['box'] for f in r]:
                iou = compute_iou(mb, rb)
                if iou > 0.3:
                    ious.append(iou)

        rows.append({
            'name': name, 'image': img,
            'mtcnn': m, 'retina': r,
            'm_count': len(m), 'r_count': len(r) if retina.available else -1,
            'm_time': m[0]['time_ms'] if m else 0,
            'r_time': r[0]['time_ms'] if r else 0,
            'm_conf': np.mean([f['confidence'] for f in m]) if m else 0,
            'r_conf': np.mean([f['confidence'] for f in r]) if r else 0,
            'iou': np.mean(ious) if ious else 0,
        })
    return rows


def print_table(rows):
    """Print comparison results as a plain text table."""
    print(f"\n{'Image':<25} {'MTCNN':>6} {'Retina':>7} {'MTCNN ms':>9} {'Retina ms':>10} "
          f"{'MTCNN Conf':>11} {'Retina Conf':>12} {'IoU':>6}")
    print("-" * 95)

    for r in rows:
        rc = str(r['r_count']) if r['r_count'] >= 0 else "N/A"
        mc = f"{r['m_conf']:.3f}" if r['m_count'] > 0 else "N/A"
        rconf = f"{r['r_conf']:.3f}" if r['r_count'] > 0 else "N/A"
        print(f"{r['name']:<25} {r['m_count']:>6} {rc:>7} {r['m_time']:>9.1f} "
              f"{r['r_time']:>10.1f} {mc:>11} {rconf:>12} {r['iou']:>6.3f}")

    total_m = sum(r['m_count'] for r in rows)
    total_r = sum(max(0, r['r_count']) for r in rows)
    avg_m_t = np.mean([r['m_time'] for r in rows])
    avg_r_t = np.mean([r['r_time'] for r in rows])
    n = len(rows)

    print("-" * 95)
    print(f"\n  {n} images tested")
    print(f"  MTCNN:     {total_m} faces, avg {avg_m_t:.1f} ms/image")
    if any(r['r_count'] >= 0 for r in rows):
        print(f"  RetinaFace: {total_r} faces, avg {avg_r_t:.1f} ms/image")
    print()


def test_preprocessing_impact(images, detector):
    """Show how preprocessing affects detection count and confidence."""
    print("\nPreprocessing impact (MTCNN):")
    print(f"  {'Image':<25} {'Without':>10} {'With':>10} {'Conf diff':>10}")
    print("  " + "-" * 58)

    for name, img in images.items():
        raw = detector.detect(img, use_preprocessing=False)
        pp = detector.detect(img, use_preprocessing=True)
        raw_c = np.mean([f['confidence'] for f in raw]) if raw else 0
        pp_c = np.mean([f['confidence'] for f in pp]) if pp else 0
        diff = pp_c - raw_c
        print(f"  {name:<25} {len(raw):>8}   {len(pp):>8}   {diff:>+.4f}")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else config.SAMPLE_IMAGES_DIR
    images = load_images(path)

    if not images:
        print(f"No images found at: {path}")
        print("Add face photos to sample_images/ or pass a path as argument.")
        return

    print(f"Loaded {len(images)} image(s)\n")

    # set up detectors
    mtcnn = MTCNNDetector()
    retina = RetinaFaceDetector()

    # run comparison
    print("Running detection comparison...")
    rows = run_comparison(images, mtcnn, retina)
    print_table(rows)

    # save comparison report
    # (redirect stdout would be cleaner but this is simpler for teammates to find)
    report_path = os.path.join(config.DATA_DIR, "detector_comparison.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        for r in rows:
            rc = str(r['r_count']) if r['r_count'] >= 0 else "N/A"
            f.write(f"{r['name']}: MTCNN={r['m_count']} faces ({r['m_time']:.0f}ms, "
                    f"conf={r['m_conf']:.3f}), RetinaFace={rc} faces ({r['r_time']:.0f}ms)\n")
    print(f"Report saved: {report_path}")

    # save side-by-side annotated images
    vis_dir = os.path.join(config.DATA_DIR, "comparison_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    for r in rows:
        mtcnn_vis = draw_boxes(r['image'], r['mtcnn'], (0, 200, 0), "MTCNN")
        if r['retina']:
            retina_vis = draw_boxes(r['image'], r['retina'], (200, 100, 0), "Retina")
            combined = np.hstack([mtcnn_vis, retina_vis])
        else:
            combined = mtcnn_vis
        cv2.imwrite(os.path.join(vis_dir, f"compare_{r['name']}"), combined)
    print(f"Visuals saved: {vis_dir}/")

    # save cropped faces for Person B
    crops_dir = os.path.join(config.DATA_DIR, "cropped_faces")
    for r in rows:
        if r['mtcnn']:
            prefix = os.path.splitext(r['name'])[0]
            save_crops(r['mtcnn'], crops_dir, prefix)
    print(f"Cropped faces saved: {crops_dir}/")

    # preprocessing test
    test_preprocessing_impact(images, mtcnn)

    print("Done. Person B can now use the cropped faces for recognition.")


if __name__ == "__main__":
    main()
