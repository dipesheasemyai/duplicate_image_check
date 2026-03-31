import os
import cv2
import numpy as np

# Inputs
query_image = "/home/easemyai/Documents/image_detection/selected_images/stove_199.jpg"
folder = "selected_images"
output_file = "pixel_results.txt"

# Load query image in grayscale
query = cv2.imread(query_image, cv2.IMREAD_GRAYSCALE)

# Resize query image to standard size
RESIZE_DIM = (256, 256)
query = cv2.resize(query, RESIZE_DIM)

def pixel_similarity(img1, img2):
    err = np.mean((img1 - img2) ** 2)
    # Convert to similarity
    similarity = 1 / (1 + err)
    return similarity

# Compare query with all images in folder
with open(output_file, "w") as f:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # Load candidate image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, RESIZE_DIM)

        # Compute similarity
        score = pixel_similarity(query, img)

        # Save only if above threshold (optional)
        THRESHOLD = 0.9
        if score >= THRESHOLD:
            line = f"{file} | Confidence: {score:.4f}\n"
            # print(line.strip())
            f.write(line)

print(f"Pixel comparison results saved to {output_file}")