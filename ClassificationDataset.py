import json
import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd

class ClassificationDataset(Dataset):

    image_files = []
    transform = None
    labels = None
    data_dir = None
    num_classes = 0

    variables = None
    with open(variables["variables_path"], "r") as f:
        variables = json.load(f)

    path_to_data = os.path.join(os.path.dirname(__file__), variables["data_root"], variables["data_path_classification"])
        

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #the image_files variable here is almost like a dictionary, where images are stored for each label
        #It is assumed that labels are also the names of files that contain corresponding images
        self.labels = os.listdir(data_dir)
        num_classes = len(self.labels)
        for label in self.labels:
            class_name = label
            for image in os.listdir(os.path.join(data_dir, class_name)):
                
                #images may need to be renamed if this dataset has not been used before
                if('_' in image):
                    os.rename(data_dir + class_name + "/" + image, data_dir + class_name + "/" + image + "_" + class_name)
                self.image_files.append(image)
        


    def __len__(self):
        return len(self.image_files)



    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.labels[idx]) + "/" + self.image_files[idx]#LEFT OFF HERE------------------make sure that path is correct for images
        img = Image.open(img_path).convert("RGB")
        #the label for classification images is the name of the outer file that contains them
        tokens = img_path.split("_")
        label = tokens[-1]

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label
    
    transform = T.Compose([
        T.Resize((192, 272)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])