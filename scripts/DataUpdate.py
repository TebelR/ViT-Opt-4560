#This script updates all the labels in the detection dataset to have their class set to 0

import os

label_dir = "../data/detection/labels" 

for split in ["train", "valid", "test"]:
    folder_path = os.path.join(label_dir, split)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = []

        for line in lines:
            tokens = line.strip().split()
            tokens[0] = '0'
            new_line = ' '.join(tokens) + '\n'
            new_lines.append(new_line)

        # Save the updated labels back
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

print("All labels updated to class 0.")