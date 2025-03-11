import os
import shutil
import random

# Set dataset directory
data_dir = "data/classification"
output_dir = "data/visionDataTemp"

# Define split sizes
train_ratio = 0.8
test_ratio = 0.2


for split in ['train', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Process each class folder
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):  # Ensure it's a directory
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle images to randomize selection
        
        # Split the dataset
        train_end = int(len(images) * train_ratio)
        
        train_images = images[:train_end]
        test_images = images[train_end:]

        # Create class folders in train, test
        for split, split_images in zip(['train', 'test'], [train_images, test_images]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img in split_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join(split_class_dir, img))
