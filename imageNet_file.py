import os
import random
import shutil

dataset_path = "/home/easemyai/.cache/kagglehub/datasets/dimensi0n/imagenet-256/versions/1"
output_folder = "selected_images"

os.makedirs(output_folder, exist_ok=True)

# get all class folders
class_folders = [f for f in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, f))]

# pick 200 classes
selected_classes = random.sample(class_folders, 200)

# choose how many classes will have 2 images
num_classes_with_two = 50  # 🔥 change this as needed

classes_with_two = set(random.sample(selected_classes, num_classes_with_two))

image_count = 0

for class_folder in selected_classes:
    class_path = os.path.join(dataset_path, class_folder)

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        continue

    # decide how many images to take
    num_images = 2 if class_folder in classes_with_two else 1

    chosen_images = random.sample(images, min(num_images, len(images)))

    for img in chosen_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_folder, f"{class_folder}_{image_count}.jpg")

        shutil.copy(src, dst)
        image_count += 1

print(f"Total images selected: {image_count}")