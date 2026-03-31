import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- CONFIG ---
folder = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train"
results_folder = "result_duplicates"
os.makedirs(results_folder, exist_ok=True)
output_txt = os.path.join(results_folder, "duplicates.txt")

RESIZE_DIM = (256, 256)
THRESHOLD = 1  # high similarity threshold for duplicate detection

# --- LOAD IMAGE ---
def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    if len(img.shape) == 2:  # grayscale -> convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, RESIZE_DIM)
    return img

# --- SIMILARITY ---
def pixel_similarity(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    return 1 - np.mean(np.abs(img1 - img2))

# --- MAIN ---
def find_duplicates(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    already_matched = set()  # skip duplicates already found

    with open(output_txt, "w") as f:
        for i, img_file in enumerate(image_files):
            if img_file in already_matched:
                continue

            img1_path = os.path.join(folder, img_file)
            img1 = load_image_rgb(img1_path)
            if img1 is None:
                continue

            for j, other_file in enumerate(image_files):
                if i >= j:  # skip self and previous comparisons
                    continue
                if other_file in already_matched:
                    continue

                img2_path = os.path.join(folder, other_file)
                img2 = load_image_rgb(img2_path)
                if img2 is None:
                    continue

                score = pixel_similarity(img1, img2)
                # Alternatively, use SSIM:
                # score, _ = ssim(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                #                 cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), full=True)

                if score >= THRESHOLD:
                    # Write only duplicates
                    f.write(f"{img_file} and {other_file} | Confidence: {score:.4f}\n")
                    # Mark duplicate so it won't be compared again
                    already_matched.add(other_file)

            print(f"Processed: {img_file}")

    print(f"\n Duplicates saved in {output_txt}")

if __name__ == "__main__":
    find_duplicates(folder)