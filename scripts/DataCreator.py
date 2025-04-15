# This script was remade a bit to generate images and label them in YAML format.
# The yaml data file contains 88 classes for the sake of the ViT. The YOLO was trained to detect a single class - the seed, so its class prediction metric is to be ignored.
# We only need to look at how the YOLO model creates bounding boxes for the seeds. The ViT model will be used to classify each cropped out seed.

#from PIL import Image
import os
import random
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np

seed_dir = "data/seedOnly"        # Directory containing seed images 
background_dir = "data/backgrounds"    # Directory containing background images
classes_dir = "data/classification"# this is used to get the class indices
# Define the output directory where the composite image will be saved.
output_dir = 'data/syntheticForTest/'
label_dir = os.path.join(output_dir, 'labels', 'test')
image_dir = os.path.join(output_dir, 'images', 'test')

os.makedirs(output_dir, exist_ok=True)   # Create the output directory if it doesn't exist
os.makedirs(label_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
# yaml_file = open(os.path.join(label_dir, "data.yaml"), "w")

min_num_seeds = 1
max_num_seeds = 100

#num_seeds = int(input("Enter the number of seeds to place in the composite image: ")) #It would be more fun to have a random number of seeds for each image

num_images_to_generate = 4# generates this amount of images with a random number of seeds per image - YAML label txt files are created for every image

seed_files = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
background_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                    if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and "l_" in f)]#added the "l_" filter to take the very large resolution backgrounds

if not seed_files:
    print("No seed images found in the seed directory.")
    exit(1)
if not background_files:
    print("No background images found in the background directory.")
    exit(1)

BACKGROUND_SIZES = {(2268, 4032), (1920, 1080), (2160, 3840)}# high resolution background sizes - these are selected randomly for each image
random.seed(16)

random_gen = random.randint(0,9999999)#this is not used as a seed, this is just to mark a generation of images produced by this script

class_indices = {}
class_names = os.listdir(classes_dir)
for i, class_name in enumerate(class_names):
    class_indices[class_name] = i

for i in range(num_images_to_generate):
    bg_file = random.choice(background_files)
    background = cv2.imread(bg_file)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    #Crop the background image to one of the high resolution sizes
    bg_width, bg_height = random.choice(list(BACKGROUND_SIZES))
    rand_x = random.randint(0, background.shape[1] - bg_width)
    rand_y = random.randint(0, background.shape[0] - bg_height)
    background = background[rand_y:rand_y+bg_height, rand_x:rand_x+bg_width]

    image_name = "image" + str(random_gen) + "_" + str(i) + ".jpg"
    label_file_name = "image" + str(random_gen) + "_" + str(i) + ".txt"
    label_file = open(os.path.join(label_dir, label_file_name), "x")
    
    for j in range(random.randint(min_num_seeds, max_num_seeds)): # Generate a random number of seeds as specified
        seed_file = random.choice(seed_files)
        seed_img = cv2.imread(seed_file, cv2.IMREAD_UNCHANGED)
        seed_img = cv2.cvtColor(seed_img, cv2.COLOR_BGRA2RGBA)
        rotate_matrix = cv2.getRotationMatrix2D(center=(seed_img.shape[1]//2, seed_img.shape[0]//2), angle=random.randint(0, 360), scale=0.5)
        cos = np.abs(rotate_matrix[0, 0])
        sin = np.abs(rotate_matrix[0, 1])
        rotated_width = int(seed_img.shape[1] * cos + seed_img.shape[0] * sin)
        rotated_height = int(seed_img.shape[1] * sin + seed_img.shape[0] * cos)
        center = (seed_img.shape[1]//2, seed_img.shape[0]//2)

        rotate_matrix[0, 2] += rotated_width/2 - center[0]
        rotate_matrix[1, 2] += rotated_height/2 - center[1]
        seed_img = cv2.warpAffine(seed_img, rotate_matrix, (rotated_width, rotated_height))
        #seed_img = seed_img[seed_img.shape[0]//4:seed_img.shape[0]//4*3, seed_img.shape[1]//4:seed_img.shape[1]//4*3, :]#crop the seed image to 1/2 of its original size
        seed_img = cv2.resize(seed_img, (seed_img.shape[1]//2, seed_img.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        seed_width, seed_height = seed_img.shape[1], seed_img.shape[0]
        #print(seed_width, seed_height)
        max_x = bg_width - seed_width
        max_y = bg_height - seed_height
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)

        alpha_channel = seed_img[:,:,3]/255.0
        alpha_inverted = 1.0 - alpha_channel
        
        seed_region = background[pos_y:pos_y+seed_height, pos_x:pos_x+seed_width]
        for c in range(3):#blend in the seed with the background based on the alpha channel
            seed_region[:,:,c] = (alpha_channel * seed_img[:,:,c] + alpha_inverted * seed_region[:,:,c])

        background[pos_y:pos_y+seed_height, pos_x:pos_x+seed_width] = seed_region

        seed_width_normalized = seed_width / bg_width
        seed_height_normalized = seed_height / bg_height
        seed_x_normalized = (pos_x+seed_width/2) / bg_width# this is the center x coordinate of the seed
        seed_y_normalized = (pos_y+seed_height/2) / bg_height # this is the center y coordinate of the seed

        seed_name_tokens =seed_file.split(".")[0].split("_")
        seed_class_name = "_".join(seed_name_tokens[1:])
        seed_class_index = class_indices[seed_class_name]

        label_file.write(f"{seed_class_index} {seed_x_normalized} {seed_y_normalized} {seed_width_normalized} {seed_height_normalized}\n")

    print("Generated image " + image_name)
        #check that normalized coordinates are accurate
        # top_left = (int((seed_x_normalized-seed_width_normalized/2) * bg_width), int((seed_y_normalized - seed_height_normalized/2)* bg_height))
        # bottom_right = (int((seed_x_normalized+seed_width_normalized/2) * bg_width), int((seed_y_normalized + seed_height_normalized/2)* bg_height))
        # cv2.rectangle(background, top_left, bottom_right, (0, 255, 0), 2)
        #cv2.putText(background, str(seed_class_index), (pos_x, pos_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        
    # plt.imshow(background)#uncomment this to see the generated image
    # plt.axis('off')
    # plt.show()

    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(image_dir, image_name), background)
    label_file.close()

