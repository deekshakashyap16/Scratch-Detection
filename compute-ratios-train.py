import os
import cv2
import csv
import numpy as np

IMAGE_DIR = r"dataset/car_dent_coco/train_denoised"
MASK_DIR = r"scratch_masks/train"
OUT_CSV = "train_ratios.csv"

image_names = []
all_ratios = []

# 1. Collect ratios (NO labels yet)
for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, file)
        mask_path = os.path.join(MASK_DIR, file)

        if not os.path.exists(mask_path):
            continue  # safety check

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        total_pixels = img.shape[0] * img.shape[1]
        scratch_pixels = np.sum(mask > 0)

        ratio = scratch_pixels / total_pixels

        image_names.append(file)
        all_ratios.append(ratio)

# 2. Compute percentile-based thresholds  ← THIS WAS MISSING
ratios = np.array(all_ratios)

low_t = np.percentile(ratios, 33)
high_t = np.percentile(ratios, 66)

def assign_severity(r):
    if r <= low_t:
        return 0   # Low
    elif r <= high_t:
        return 1   # Medium
    else:
        return 2   # High

# 3. Assign severity labels
rows = []
for name, r in zip(image_names, ratios):
    severity = assign_severity(r)
    rows.append([name, r, severity])

unique, counts = np.unique([r[2] for r in rows], return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# 5. Save CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "scratch_ratio", "severity"])
    writer.writerows(rows)

print("Saved:", OUT_CSV)
