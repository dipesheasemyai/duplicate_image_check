import os
import cv2
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
results_folder = "result_test_final1"
os.makedirs(results_folder, exist_ok=True)

RESIZE_DIM = (256, 256)

SSIM_THRESHOLD = 0.95 

# --- ADVANCED SIMILARITY (SSIM) ---
def get_similarity_score(img1, img2):
    # 1. Convert to grayscale for structural comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    

    score, _ = ssim(gray1, gray2, full=True)
    
    
    if score > 0.8:
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0], None, [180], [0, 180])
        hist2 = cv2.calcHist([hsv2], [0], None, [180], [0, 180])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        color_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # We average them or require both to be high
        return (score + color_score) / 2
    
    return score

# --- LOAD IMAGE AS BGR ---
def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    # Ensure it's 3-channel BGR even if source is grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, RESIZE_DIM)
    return img

# --- COPY IMAGE TO RESULTS FOLDER ---
def copy_image_to_result(src_path):
    dst_path = os.path.join(results_folder, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    return os.path.basename(src_path)

# --- MAIN PROCESS ---
def image_process(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    html_file = os.path.join(results_folder, "all_comparisons.html")

    with open(html_file, "w") as f:
        f.write("""
        <html>
        <head>
            <style>
                table {border-collapse: collapse; width: 100%; font-family: sans-serif;}
                th, td {border: 1px solid #ddd; padding: 12px; text-align: center;}
                tr:nth-child(even){background-color: #f2f2f2;}
                img {width: 200px; border-radius: 4px;}
                .score-high { color: green; font-weight: bold; }
            </style>
        </head>
        <body>
            <h2>Advanced Image Comparison Results</h2>
            <table>
                <tr>
                    <th>Image 1</th>
                    <th>Image 2</th>
                    <th>Similarity Score (SSIM + Color)</th>
                </tr>
        """)

        for i, img_file in enumerate(image_files):
            img1_path = os.path.join(folder, img_file)
            img1 = load_image_rgb(img1_path)
            if img1 is None: continue

            for j, other_file in enumerate(image_files):
                if j <= i: continue 

                img2_path = os.path.join(folder, other_file)
                img2 = load_image_rgb(img2_path)
                if img2 is None: continue

                score = get_similarity_score(img1, img2)

                # Only log if they are genuinely similar based on our new threshold
                if score >= SSIM_THRESHOLD:
                    img1_display = copy_image_to_result(img1_path)
                    img2_display = copy_image_to_result(img2_path)

                    f.write(f"""
                    <tr>
                        <td><img src="{img1_display}"><br>{img_file}</td>
                        <td><img src="{img2_display}"><br>{other_file}</td>
                        <td class="score-high">{score:.4f}</td>
                    </tr>
                    """)

            print(f"Checked: {img_file}")

        f.write("</table></body></html>")

    print(f"\nProcess Complete. Results here: {html_file}")

if __name__ == "__main__":
    # Update this path to your local directory
    target_folder = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train"
    image_process(target_folder)