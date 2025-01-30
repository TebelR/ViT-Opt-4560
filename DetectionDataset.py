# import torch
# from torch.utils.data import Dataset
# import os
# from torchvision.io import read_image

# class DetectionDataset(Dataset):
#     img_dir = None
#     label_dir = None
#     img_list = None
#     label_list = None

#     def __init__(self, img_dir, label_dir, transform=None):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.img_list = os.listdir(self.img_dir)
#         self.label_list = os.listdir(self.label_dir)

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_list[idx])
#         image = read_image(img_path)
#         with open(os.path.join(self.label_dir, self.label_list[idx]), 'r') as f:
#             label = torch.tensor([float(x) for x in f.read().split()])
#         if self.transform:
#             image = self.transform(image)
#         return image, label

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
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
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
    
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

