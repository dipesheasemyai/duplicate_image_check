import cv2
import numpy as np

image1 = "/home/easemyai/Documents/image_detection/selected_images/badger_44.jpg"

image2 = "/home/easemyai/Documents/image_detection/selected_images/badger_45.jpg"

gray_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
gray_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

gray_img1_resize = cv2.resize(gray_img1, (256, 256))
gray_img2_resize = cv2.resize(gray_img2, (256, 256))

# similarity = 1 - (np.mean(np.abs(gray_img1_resize - gray_img2_resize)) / 255)
# print(similarity)

import cv2
import numpy as np

def mse(imageA, imageB):
    # Ensure the images have the same size
    assert imageA.shape == imageB.shape, "Images must be the same size."
    
    # Calculate the MSE between the images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err

# Load images
imageA = cv2.resize(gray_img1, (256, 256))
imageB = cv2.resize(gray_img1, (256, 256))

# Compute MSE
error = mse(imageA, imageB)
print(f"Mean Squared Error: {error}")