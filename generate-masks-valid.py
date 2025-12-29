import os
import cv2
import numpy as np

INPUT_DIR = r"dataset/car_dent_coco/car_dent_coco/valid"
OUTPUT_DIR = r"scratch_masks/valid"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_scratch_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance thin scratches
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Strengthen scratches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Binary mask
    _, mask = cv2.threshold(dilated, 1, 255, cv2.THRESH_BINARY)

    return mask

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(INPUT_DIR, file)
        img = cv2.imread(img_path)

        mask = extract_scratch_mask(img)

        out_path = os.path.join(OUTPUT_DIR, file)
        cv2.imwrite(out_path, mask)

        print("Saved mask:", file)

print("DONE")
