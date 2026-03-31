from skimage.metrics import structural_similarity as ssim
import cv2
import os

# Load images in grayscale
image1 = "/home/easemyai/Documents/image_detection/selected_images/cat_1.jpeg"

# image2 = "/home/easemyai/Documents/image_detection/selected_images/cat_2.jpg"

# gray_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
# gray_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

# gray_img1_resize = cv2.resize(gray_img1, (256, 256))
# gray_img2_resize = cv2.resize(gray_img2, (256, 256))

# # Compute SSIM (data_range inferred for uint8 images)
# score, diff = ssim(gray_img1_resize, gray_img2_resize, full=True, data_range=255)
# print(f"SSIM Score: {score}")

folder = "selected_images"
output_file = "scikit_image_results.txt"

# Load query image in grayscale
query = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)

# Resize query image to standard size
RESIZE_DIM = (256, 256)
query = cv2.resize(query, RESIZE_DIM)

def pixel_similarity(img1, img2):
    score, diff = ssim(img1, img2, full=True, data_range=255)
    return score

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