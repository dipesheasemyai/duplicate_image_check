import cv2
import os 
import numpy as np


folder_path = '/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train'
result_folder = 'duplicate_image_folder'
os.makedirs(result_folder, exist_ok=True)
output_txt = os.path.join(result_folder, "result_duplicate_image_list.txt")

resize_dim = (256, 256)
threshold = 1

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize_dim)
    return img

def compare_image_similarity(img1, img2):
    error = np.mean((img1 - img2)**2)
    return 1 / ( 1 + error)


def is_copy(filename):
    lower = filename.lower()

    if 'copy' in lower or 'duplicate' in lower or '(' in lower:
        return True
    return False

def has_jpg_hint(filename):
    return "_jpg" in filename.lower()

def has_png_hint(filename):
    return "_png" in filename.lower()

def choose_original(group, folder_path):
    clean_files = []
    for f in group:
        if not is_copy(f):
            clean_files.append(f)

    candiates = clean_files if clean_files else group

    jpg_files = []
    for f in candiates:
        if f.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.append(f)
    
    png_files = []
    for f in candiates:
        if f.lower().endswith(('.png')):
            png_files.append(f)

    for f in candiates:
        if has_jpg_hint(f) and jpg_files:
            return min(jpg_files, key=len)
        
    for f in candiates:
        if has_png_hint(f) and png_files:
            return min(png_files, key=len)
        

    if jpg_files and png_files:
        jpg = jpg_files[0]
        png = png_files[0]

        jpg_size = os.path.getsize(os.path.join(folder_path, jpg))
        png_size = os.path.getsize(os.path.join(folder_path, png))

        if png_size > jpg_size * 1.2:
            return png
        
        return jpg


    if jpg_files:
        return min(jpg_files, key=len)
    
    if png_files:
        return min(png_files, key=len)

    return min(candiates, key=len)


def find_duplicate(folder):
    image_file = []

    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_file.append(f)

    already_match = set()
    duplicate_dict = {}

    for i, img_file in enumerate(image_file):
        if img_file in already_match:
            continue

        img1_path = os.path.join(folder, img_file)
        img1 = load_image(img1_path)
        if img1 is None:
            continue

        group = [img_file]

        for j , other_file in enumerate(image_file):
            if i ==j or  other_file in already_match:
                continue

            img2_path = os.path.join(folder, other_file)
            img2 = load_image(img2_path)
            if img2 is None:
                continue
                
            score = compare_image_similarity(img1, img2)

            if score >= threshold:
                group.append(other_file)
                already_match.add(other_file)

        
        if len(group) > 1:
            original = choose_original(group, folder)
            copies = []
            for f in group:
                if f != original:
                    copies.append(f)
            duplicate_dict[original] = copies
        
        already_match.add(img_file)

    
    with open(output_txt, 'w') as f:
        for original, copies in duplicate_dict.items():
            line = f"{original} -> {', '.join(copies)}\n"
            f.write(line)

    print(f"\n Duplicates saved in {output_txt}")

if __name__ == "__main__":
    find_duplicate(folder_path)

 



