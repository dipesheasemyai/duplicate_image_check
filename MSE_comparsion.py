import cv2
import numpy as np

image1 = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train/00103_jpg.rf.9d964e1c1ab2ecd4f0a8535fd738507e.jpg"

image2 = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train/00103_jpg.rf.853bf8ed4a919a8d72bb58c8d3c3f5d3.jpg"

gray_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
gray_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

gray_img1_resize = cv2.resize(gray_img1, (256, 256))
gray_img2_resize = cv2.resize(gray_img2, (256, 256))


import cv2
import numpy as np

def mse(imageA, imageB):
    # Ensure the images have the same size
    err = np.mean((imageA - imageB) ** 2)
    return 1 / (1 + err)

# Load images
imageA = cv2.resize(gray_img1, (256, 256))
imageB = cv2.resize(gray_img1, (256, 256))

# Compute MSE
error = mse(imageA, imageB)
print(f"Mean Squared Error: {error}")