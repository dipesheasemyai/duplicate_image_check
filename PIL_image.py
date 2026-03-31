from PIL import Image
import numpy as np


image1 = "safety_ai.v1i.coco/train/Video4_8_jpg.rf.0dcd2409028917ad27d844d74e7fdbfd.jpg"

image2 = "safety_ai.v1i.coco/train/Video4_8_jpg.rf.02d6b98f9ee15effcebe9dad5d0d2722 (Copy 2).jpg"
# Compare two images pixel by pixel
img1 = np.array(Image.open(image1))
img2 = np.array(Image.open(image2))

# Check if the shapes are the same and pixels are identical
if img1.shape == img2.shape and np.all(img1 == img2):
    print("Images are identical!")
else:
    print("Images are different.")