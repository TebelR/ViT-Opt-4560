#This is a custom dataset for the ultralytics YOLO detection model.

import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd

class DetectionDataset(Dataset):
    image_dir = None
    label_dir = None
    image_files = None

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load label file
        label_file = img_path.with_suffix('.txt').replace(self.image_dir, self.label_dir)
        label_data = pd.read_csv(label_file, delimiter=' ', header=None)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, label_data.to_numpy()
    
    #default transform function for the dataset will resize images to 640x640 and normalize them since YAML bounding box labels are normalized
    #YOLO expects the labels to follow the format [class, x_center, y_center, width, height] where the last 4 parameters are normalized
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

