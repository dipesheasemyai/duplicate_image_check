# import os
import cv2
# import numpy as np

image1 = "/home/easemyai/Documents/image_detection/selected_images/cat_1.jpeg"

image2 = "/home/easemyai/Documents/image_detection/selected_images/cat_1.jpeg"

gray_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
gray_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

gray_img1_resize = cv2.resize(gray_img1, (256, 256))
gray_img2_resize = cv2.resize(gray_img2, (256, 256))

# Compute histogram
hist1 = cv2.calcHist([gray_img1_resize], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray_img2_resize], [0], None, [256], [0, 256])

hist1 = cv2.normalize(hist1, hist1).flatten()
hist2 = cv2.normalize(hist2, hist2).flatten()

# Compare histogram
methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA']

for method in methods:
    score = cv2.compareHist(hist1, hist2, getattr(cv2, f'HISTCMP_{method}'))
    print(f"{method} comparsion socre: {score}")
