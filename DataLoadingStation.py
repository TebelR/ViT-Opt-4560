#This class will load data from the data directory and function as a runtime database of images.
import torch
from torch.utils.data import DataLoader
import os
from DetectionDataset import DetectionDataset, random_split
class DataLoadingStation:
    data_root = "../data/"
    dataset_detection = None

    dataset_train_detection = None
    dataset_validate_detection = None
    dataset_test_detection = None

    TRAIN_SPLIT = 0.8
    VALIDATE_SPLIT = 0.1
    TEST_SPLIT = 0.1

    BATCH_SIZE = 2
    NUM_WORKERS = 0

    dl_train_detection = None
    dl_validate_detection = None
    dl_test_detection = None

    def __init__(self):
        self.load_data_detection(self.data_root)

    
    #Loads images from the data directory defined by the path argument.
    #The DetectionDataset is a custom class that extends torch.utils.data.Dataset, it contains both images and labels for detection.
    #Originally, the dataset was following a YAML format, but this setup does not utilize that. Custom datasets are used instead, just in case we need to add more data later.
    def load_data_detection(self):
        self.dataset_detection = DetectionDataset(self.data_root + "detection/images/", self.data_root + "detection/labels/")
        train_size = int(len(self.dataset_detection) * self.TRAIN_SPLIT)
        validate_size = int(len(self.dataset_detection) * self.VALIDATE_SPLIT)
        test_size = int(len(self.dataset_detection) * self.TEST_SPLIT)

        self.dataset_train_detection, self.dataset_validate_detection, self.dataset_test_detection = random_split(self.dataset_detection, [train_size, validate_size, test_size])
        print("Detection dataset loaded.")
        print("Train size:", len(self.dataset_train_detection))
        print("Validate size:", len(self.dataset_validate_detection))
        print("Test size:", len(self.dataset_test_detection))

        self.dl_train_detection = DataLoader(self.dataset_train_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.dl_validate_detection = DataLoader(self.dataset_validate_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.dl_test_detection = DataLoader(self.dataset_test_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

        print("Detection dataloaders created.")