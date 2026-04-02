import os
import cv2
import numpy as np



def load_image(path):
    RESIZE_DIM = (256, 256)
    img = cv2.imread(path)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, RESIZE_DIM)
    return img


def compute_similarity(img1, img2):
    score = np.mean((img1 - img2) ** 2)
    return 1 / (1 + score)


def is_copy(filename):
    lower = filename.lower()
    if "(" in lower and ")" in lower:
        return True
    if "copy" in lower or "duplicate" in lower or "_copy" in lower:
        return True
    return False


def has_jpg_hint(filename):
    return "_jpg" in filename.lower()

def has_png_hint(filename):
    return "_png" in filename.lower()

def choose_original(group, folder_path):
    clean_files = [f for f in group if not is_copy(f)]
    candidates = clean_files if clean_files else group

    # 2. Separate formats
    png_files = [f for f in candidates if f.lower().endswith(".png")]
    jpg_files = [f for f in candidates if f.lower().endswith((".jpg", ".jpeg"))]

    for f in candidates:
        if has_jpg_hint(f) and jpg_files:
            return min(jpg_files, key=len)
    
    for f in candidates:
        if has_png_hint(f) and png_files:
            return min(png_files, key=len)
    
    if png_files and jpg_files:
        png = png_files[0]
        jpg = jpg_files[0]

        png_size = os.path.getsize(os.path.join(folder_path, png))
        jpg_size = os.path.getsize(os.path.join(folder_path, jpg))

        if png_size > jpg_size * 1.2:
            return png
        
        return jpg
    
    if png_files:
        return min(png_files, key=len)
    
    if jpg_files:
        return min(jpg_files, key=len)

    return min(candidates, key=len)


# --- MAIN FUNCTION ---
def find_duplicates(folder):

    results_folder = "result_image_duplicate"
    os.makedirs(results_folder, exist_ok=True)
    output_txt = os.path.join(results_folder, "duplicates_image_list.txt")

    THRESHOLD = 1
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    already_matched = set()
    duplicates_dict = {}

    for i, img_file in enumerate(image_files):
        if img_file in already_matched:
            continue

        img1_path = os.path.join(folder, img_file)
        img1 = load_image(img1_path)
        if img1 is None:
            continue

        group = [img_file]

        for j, other_file in enumerate(image_files):
            if i == j or other_file in already_matched:
                continue

            img2_path = os.path.join(folder, other_file)
            img2 = load_image(img2_path)
            if img2 is None:
                continue

            score = compute_similarity(img1, img2)

            if score >= THRESHOLD:
                group.append(other_file)
                already_matched.add(other_file)

        # Process group
        if len(group) > 1:
            original = choose_original(group, folder)
            copies = [f for f in group if f != original]
            duplicates_dict[original] = copies

        already_matched.add(img_file)
        # print(f"Processed: {img_file}")

    # --- SAVE RESULTS ---
    with open(output_txt, "w") as f:
        for original, copies in duplicates_dict.items():
            line = f"{original} -> {', '.join(copies)}\n"
            f.write(line)

    print(f"\n Duplicates saved in: {output_txt}")


