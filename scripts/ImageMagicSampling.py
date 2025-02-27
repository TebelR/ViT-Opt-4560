import os
from collections import defaultdict
import shutil
dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "classification")


class_counts = defaultdict(int)


for class_name in os.listdir(dataset_path):
    for image_name in os.listdir(os.path.join(dataset_path, class_name)):
        class_counts[class_name] += 1


print(class_counts)


target_count = 106

# Compute how many extra images are needed per class
needed_images = {cls: target_count - count for cls, count in class_counts.items() if count < target_count}

print(needed_images)



# duplicate images
for cls, needed in needed_images.items():
    class_images = [f for f in os.listdir(dataset_path) if f.endswith(f"_{cls}.jpg")]
    
    for i in range(needed):
        original_img = class_images[i % len(class_images)] 
        new_img_name = f"changed{i+10000}_{cls}.jpg"
        shutil.copy(os.path.join(dataset_path, original_img), os.path.join(dataset_path, new_img_name))

print("Dataset balancing complete!")