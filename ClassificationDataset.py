import json
import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy
import torch
# import pandas as pd
import torchvision.io as io

class ClassificationDataset(Dataset):

    image_files = []
    transform = None
    labels = None
    data_dir = None
    num_classes = 0
    indexed_classes = {}
    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)

    path_to_data = os.path.join(os.path.dirname(__file__), variables["data_root"], variables["data_path_classification"])
        
    

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #the image_files variable here is almost like a dictionary, where images are stored for each label
        #It is assumed that labels are also the names of files that contain corresponding images
        self.labels = os.listdir(data_dir)
        self.num_classes = len(self.labels)

        for label in self.labels:
            self.indexed_classes[label] = self.labels.index(label)

            for image in os.listdir(os.path.join(data_dir, label)):
                image_name = image.split(".")[0].split("_")[0]
                img_name_reformatted = image_name + "_" + label + ".jpg"
                #os.rename(os.path.join(data_dir, label, image), os.path.join(data_dir, label, img_name_reformatted)) #uncomment if data looks strange
                self.image_files.append(img_name_reformatted)
        


    def __len__(self):
        return len(self.image_files)



    def __getitem__(self, idx):
        #img_path = os.path.join(self.data_dir, self.labels[idx]) + "/" + self.image_files[idx]
        image_f_name = self.image_files[idx]
        image_name_only = image_f_name.split(".")[0]#triple curse
        image_class_substring = image_name_only.split("_")[1:]
        image_class = "_".join(image_class_substring)
        img_path = os.path.join(self.data_dir, image_class, image_f_name)
        #img = Image.open(img_path).convert("RGB")
        img = io.decode_image(img_path, mode="RGB")
        #img = numpy.array(img)
        #the label for classification images is the name of the outer file that contains them
        # tokens = img_path.split("_")
        # label = tokens[-1]

        # Apply transformations
        #if self.transform:
        transform = T.Compose([
            T.Resize((192, 272)),
            #T.ToTensor(),
            # T.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # T.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # T.RandomErasing(p=0.3),
            
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img.to(torch.float32)/255.0)
        #img = numpy.array(img)

        return img, self.indexed_classes[image_class]
    
    