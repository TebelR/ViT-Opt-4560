import os
from DataLoadingStation import DataLoadingStation
import torch
import torchvision
import random
from torchvision import io
from tqdm import tqdm

data_path = "data/syntheticForViT"
transformation_path = "data/classificationSynthetic"

#this is the chained dictionary for classes 0-87, this will have each index filled with 400 images at some point
seed_images = {}

dls = DataLoadingStation()
dls.load_data_classification(0.9, 0.1001)
class_dict = dls.dataset_classification.indexed_classes

os.makedirs(transformation_path, exist_ok=True)
#infer on a bunch of random images, detect the seeds and then using the appropriate labels, crop out the seeds
#store the seeds to then fill up 88 classes with 400 images per class




def fillImagesDict():
        
    all_images = os.listdir(os.path.join(data_path, "images", "test"))
    all_labels = os.listdir(os.path.join(data_path, "labels", "test"))
        
    for i in tqdm(range(len(all_images)), desc = "Traversing Images", unit = "image"):
        image_path = os.path.join(data_path, "images", "test", all_images[i])
        label_path = os.path.join(data_path, "labels", "test", all_labels[i])
            
        true_box_classes = torch.tensor([list(map(float, line.split())) for line in open(label_path)], dtype=torch.float32)
        
        for class_box in true_box_classes:
            image = io.decode_image(image_path, mode="RGB")
            #take every seed with its true box and crop it out of the big-res image
            #the images in labels follow the format [class, x, y, width, height] - x y width height are normalized w respect to totlal image size

            x = class_box[1] * image.shape[2]
            y = class_box[2] * image.shape[1]
            w = class_box[3] * image.shape[2]
            h = class_box[4] * image.shape[1]
            seed = image[:, int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            class_index = int(class_box[0])

            append_images_dict(seed, class_index)
    
    scanAndDuplicate()

    

def append_images_dict(image, class_index):
    if(len(seed_images[class_index]) < 400):
        seed_images[class_index].append((image.to(torch.float32)/255.0).unsqueeze(0))


#look at the current state of the seed images for each class and duplicate images if needed to fill up the quota (400 images per class)
def scanAndDuplicate():
    print("Reached the end of training images, scanning and duplicating lacking class images to meet the quota...")
    for key in seed_images:
        appended_images = 0
        while(len(seed_images[key]) < 400):
            seed_images[key].append(seed_images[key][random.randint(0, len(seed_images[key])-1)])
            appended_images+=1
        print(f"Fixed up class {key} with {appended_images} images")
    print("Fixed up all classes")
        
    
def saveImageFiles():
    reverse_dict = {v: k for k, v in class_dict.items()}
    for value in seed_images:
        for i in range(len(seed_images[value])):
            class_name = reverse_dict[value]
            image = seed_images[value][i]
            torchvision.utils.save_image(image, os.path.join(transformation_path, class_name, class_name + "_" + str(i) + ".png"))
            





#execution code below----------------------------------------------

for key, value in class_dict.items():
    os.makedirs(os.path.join(transformation_path, str(key)), exist_ok=True)
    seed_images[value] = []

fillImagesDict()
for key, value in seed_images.items():
    print("Class " + str(key) + " has " + str(len(value)) + " images")
print("Checkpoint")
saveImageFiles()