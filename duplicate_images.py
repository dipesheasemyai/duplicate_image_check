import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- CONFIG ---
folder = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train"
results_folder = "result_duplicates"
os.makedirs(results_folder, exist_ok=True)
output_txt = os.path.join(results_folder, "duplicates_grouped.txt")

RESIZE_DIM = (256, 256)
THRESHOLD = 0.95  # similarity threshold to consider as duplicate

# --- LOAD IMAGE ---
def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    if len(img.shape) == 2:  # grayscale -> convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, RESIZE_DIM)
    return img

# --- COMPUTE SIMILARITY ---
def compute_similarity(img1, img2):
    # Convert to grayscale for SSIM comparison
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

# --- MAIN DUPLICATE FINDER ---
def find_grouped_duplicates(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    already_matched = set()  # skip duplicates already recorded

    grouped_duplicates = {}  # store original -> list of duplicates

    for i, img_file in enumerate(image_files):
        if img_file in already_matched:
            continue

        img1_path = os.path.join(folder, img_file)
        img1 = load_image_rgb(img1_path)
        if img1 is None:
            continue

        duplicates = []

        for j, other_file in enumerate(image_files):
            if i >= j:  # skip self and previous comparisons
                continue
            if other_file in already_matched:
                continue

            img2_path = os.path.join(folder, other_file)
            img2 = load_image_rgb(img2_path)
            if img2 is None:
                continue

            score = compute_similarity(img1, img2)

            if score >= THRESHOLD:
                duplicates.append(other_file)
                already_matched.add(other_file)

        if duplicates:
            grouped_duplicates[img_file] = duplicates

        print(f"Processed: {img_file}")

    # --- SAVE RESULTS ---
    with open(output_txt, "w") as f:
        for original, copies in grouped_duplicates.items():
            line = f"{original} -> {', '.join(copies)}\n"
            f.write(line)

    print(f"\n Grouped duplicates saved in {output_txt}")

if __name__ == "__main__":
    find_grouped_duplicates(folder)