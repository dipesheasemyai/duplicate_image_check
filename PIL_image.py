from PIL import Image
import numpy as np


image1 = "/home/easemyai/Documents/image_detection/selected_images/chiffonier_172.jpg"

image2 = "/home/easemyai/Documents/image_detection/selected_images/chiffonier_173.jpg"
# Compare two images pixel by pixel
img1 = np.array(Image.open(image1))
img2 = np.array(Image.open(image2))

# Check if the shapes are the same and pixels are identical
if img1.shape == img2.shape and np.all(img1 == img2):
    print("Images are identical!")
else:
    print("Images are different.")