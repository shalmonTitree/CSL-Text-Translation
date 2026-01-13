#!/usr/bin/env python3
"""
FAST MediaPipe Holistic feature extraction for PHOENIX-2014-T

Optimizations included:
✅ Skip frames (temporal stride)
✅ Disable face landmarks
✅ Lower model complexity
✅ Resize frames
✅ Multi-process CPU execution

Output:
  one .npy file per sample: (T, 225)
"""

import os
import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ==========================
# CONFIG (SAFE DEFAULTS)
# ==========================
FRAME_STRIDE = 3          # keep every 3rd frame (25fps → ~8fps)
IMG_SIZE = 256            # resize frames
MODEL_COMPLEXITY = 0      # fastest
NUM_WORKERS = 6           # adjust based on CPU (6–8 ideal)

# ==========================
# MEDIAPIPE IMPORT (LOCAL)
# ==========================
def init_mediapipe():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features_from_image(img_rgb, holistic):
    """
    Output vector: 225 dims
    - Pose: 33 × 3
    - LH:   21 × 3
    - RH:   21 × 3
    """
    res = holistic.process(img_rgb)
    feat = []

    # Pose
    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 33 * 3)

    # Left hand
    if res.left_hand_landmarks:
        for lm in res.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 21 * 3)

    # Right hand
    if res.right_hand_landmarks:
        for lm in res.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 21 * 3)

    return np.array(feat, dtype=np.float32)

# ==========================
# PROCESS ONE SAMPLE
# ==========================
def process_one_sample(args):
    sample_folder, out_dir = args
    sample_id = os.path.basename(sample_folder)
    out_path = os.path.join(out_dir, sample_id + ".npy")

    if os.path.exists(out_path):
        return

    holistic = init_mediapipe()

    # Load frames
    frames = sorted(
        glob.glob(os.path.join(sample_folder, "*.png")) +
        glob.glob(os.path.join(sample_folder, "*.jpg"))
    )[::FRAME_STRIDE]

    feats = []
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        feats.append(extract_features_from_image(img, holistic))

    holistic.close()

    if len(feats) == 0:
        print(f"[WARN] No frames in {sample_id}")
        return

    np.save(out_path, np.stack(feats))


# ==========================
# MAIN
# ==========================
def main(frames_root, out_root, split):
    split_root = os.path.join(frames_root, split)
    assert os.path.isdir(split_root), f"{split_root} not found"

    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    sample_folders = sorted([
        os.path.join(split_root, d)
        for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    ])

    args = [(folder, out_dir) for folder in sample_folders]

    print(f"Extracting {split}: {len(sample_folders)} samples")
    print(f"Frame stride = {FRAME_STRIDE}, Workers = {NUM_WORKERS}")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(
            tqdm(
                executor.map(process_one_sample, args),
                total=len(args),
                desc=f"Extracting {split}"
            )
        )

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", required=True,
                        help="Root containing train/dev/test frame folders")
    parser.add_argument("--out_root", default="data/features",
                        help="Output directory for .npy files")
    parser.add_argument("--split", required=True,
                        choices=["train", "dev", "test"])
    args = parser.parse_args()

    main(args.frames_root, args.out_root, args.split)
