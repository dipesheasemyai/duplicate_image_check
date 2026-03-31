import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import cosine_similarity

# Load model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feature = model(img)

    return feature.flatten()

query_image = "/home/easemyai/Documents/image_detection/selected_images/cat_2.jpg"
folder = "selected_images"
output_file = "results.txt"

query_feature = extract_feature(query_image)

THRESHOLD = 0.8

# open file to write
with open(output_file, "w") as f:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        feat = extract_feature(path)

        score = cosine_similarity(query_feature.unsqueeze(0),
                                  feat.unsqueeze(0)).item()

        if score > THRESHOLD:
            line = f"{file} | Confidence: {score:.4f}\n"
            print(line.strip())   # optional (also print)
            f.write(line)

print(f"Results saved to {output_file}")