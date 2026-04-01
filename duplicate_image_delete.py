import os 

folder = "/home/easemyai/Documents/image_detection/safety_ai.v1i.coco/train"
results_folder = "result_duplicates_final1/duplicates_only.txt"

to_delete = []

with open(results_folder, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or "->" not in line:
            continue
        parts = line.split("->")
        copies = []
        for c in parts[1].split(","):
            c = c.strip()
            if c:
                copies.append(c)
        to_delete.extend(copies)


deleted_count = 0
for filename in to_delete:
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        deleted_count += 1
        print(f"Deleted: {filename}")
    else:
        print(f"File not found, skipping :{filename}")

print(f"\n Total deleted : {deleted_count}")

        