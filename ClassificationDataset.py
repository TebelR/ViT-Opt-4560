import json
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
# import pandas as pd
import torchvision.io as io


#This is a custom dataset for the ViT classification model
#Originally this was intended to work just with the dataset of 88 different seed classes, but it has since been reworked to work with the synthetic dataset
#For the synthetic classification dataset, seeds were just croppted out of large synthetic images - this is what was used to train the ViT at the end.
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

    path_to_data = os.path.join(os.path.dirname(__file__), "data", "classificationSynthetic")#this is for synthetic data training
    

    def __init__(self, data_dir, transform=None):
        self.data_dir = self.path_to_data#data_dir
        self.transform = transform
        #the image_files variable here is almost like a dictionary, where images are stored for each label
        #It is assumed that labels are also the names of files that contain corresponding images
        self.labels = os.listdir(self.data_dir)
        self.num_classes = len(self.labels)

        for label in self.labels:
            self.indexed_classes[label] = self.labels.index(label)

            for image in os.listdir(os.path.join(self.data_dir, label)):
                self.image_files.append(image)
        


    def __len__(self):
        return len(self.image_files)


#this is the getItem for the synthetic cropped-out seeds
    def __getitem__(self, idx):

        image_f_name = self.image_files[idx]
        image_name_only = image_f_name.split(".")[0]
        image_class_substring = image_name_only.split("_")[:-1]
        image_class = "_".join(image_class_substring)
        img_path = os.path.join(self.data_dir, image_class, image_f_name)
        img = io.decode_image(img_path, mode="RGB")

        #The normalization values were acquired from inspecting the data distribution of the synthetic dataset
        transform = T.Compose([
            T.Resize((80,80)),
            T.Normalize(mean=[0.634, 0.562, 0.498], std=[0.204, 0.241, 0.244])
        ])
        img = transform(img.to(torch.float32)/255.0)

        return img, self.indexed_classes[image_class]
    
    