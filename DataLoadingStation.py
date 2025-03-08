#This class will load data from the data directory and function as a runtime database of images.
import torch
from torch.utils.data import DataLoader
import os

import torchvision
from DetectionDataset import DetectionDataset
from ClassificationDataset import ClassificationDataset
from torch.utils.data import random_split
import json
from torchvision import transforms as T

class DataLoadingStation:
    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)


    project_root = os.path.dirname(__file__)
    data_root = os.path.join(project_root,  variables["data_root"])
    detection_path = os.path.join(data_root, variables["data_path_detection"])
    classification_path = os.path.join(data_root, variables["data_path_classification"])
    dataset_detection = None

    dataset_train_detection = None
    dataset_validate_detection = None
    dataset_test_detection = None

    dataset_train_classification = None
    dataset_validate_classification = None
    dataset_test_classification = None

    #default splits for training, validation, and testing
    TRAIN_SPLIT_DETECTION = 0.8
    VALIDATE_SPLIT_DETECTION = 0.1
    TEST_SPLIT_DETECTION = 0.1

    TRAIN_SPLIT_CLASSIFICATION = 0.7
    VALIDATE_SPLIT_CLASSIFICATION = 0.1001
    TEST_SPLIT_CLASSIFICATION = 0.2

    BATCH_SIZE = 16
    NUM_WORKERS = 0

    num_classes = 0

    dataset_classification = None

    dl_train_detection = None
    dl_validate_detection = None
    dl_test_detection = None

    dl_train_classification = None
    dl_validate_classification = None
    dl_test_classification = None

    def __init__(self):
        pass

    
    #Loads images from the data directory defined by the path argument.
    #The DetectionDataset is a custom class that extends torch.utils.data.Dataset, it contains both images and labels for detection.
    #Originally, the dataset was following a YAML format, but this setup does not utilize that. Custom datasets are used instead, just in case we need to add more data later.
    def load_data_detection(self, training_split, validation_split, test_split):
        self.TRAIN_SPLIT_DETECTION = training_split
        self.VALIDATE_SPLIT_DETECTION = validation_split
        self.TEST_SPLIT_DETECTION = test_split
        self.dataset_detection = DetectionDataset(os.path.join(self.detection_path, "images"), os.path.join(self.detection_path, "labels"))
        train_size = int(len(self.dataset_detection) * self.TRAIN_SPLIT_DETECTION)
        validate_size = int(len(self.dataset_detection) * self.VALIDATE_SPLIT_DETECTION)
        test_size = int(len(self.dataset_detection) * self.TEST_SPLIT_DETECTION)

        self.dataset_train_detection, self.dataset_validate_detection, self.dataset_test_detection = random_split(self.dataset_detection, [train_size, validate_size, test_size])
        print("Detection dataset loaded.")
        print("Train size:", len(self.dataset_train_detection))
        print("Validate size:", len(self.dataset_validate_detection))
        print("Test size:", len(self.dataset_test_detection))
        print("Total size:", len(self.dataset_detection))

        self.dl_train_detection = DataLoader(self.dataset_train_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.dl_validate_detection = DataLoader(self.dataset_validate_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.dl_test_detection = DataLoader(self.dataset_test_detection, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

        print("Detection dataloaders created.")
    #ALL OF THE ABOVE CODE MIGHT NOT BE NECESSARY, YOLOV11 TRAINS AND DOES INFERENCE FROM YAML DATA




    #This will make one giant dataset of images with respective labels for classification - no need to divide the dataset into train, test, validate.
    #The dataset is then randomly split based on arguments into train, validate, and test datasets.
    def load_data_classification(self, training_split, validation_split):#, test_split):
        self.TRAIN_SPLIT_CLASSIFICATION = training_split
        self.VALIDATE_SPLIT_CLASSIFICATION = validation_split
        #self.TEST_SPLIT_CLASSIFICATION = test_split

        self.dataset_classification = ClassificationDataset(self.classification_path)
        self.num_classes = self.dataset_classification.num_classes
        train_size = int(len(self.dataset_classification) * self.TRAIN_SPLIT_CLASSIFICATION)
        validate_size = int(len(self.dataset_classification) * self.VALIDATE_SPLIT_CLASSIFICATION)
        #test_size = int(len(self.dataset_classification) * self.TEST_SPLIT_CLASSIFICATION)
        print("Train size:", (train_size))
        print("Validate size:", (validate_size))
        #print("Test size:", (test_size))
        print("Total size:", len(self.dataset_classification))

        self.dataset_train_classification, self.dataset_validate_classification = random_split(self.dataset_classification, [train_size, validate_size])#, test_size]) , self.dataset_test_classification
        print("Classification dataset loaded.")
        # print("Train size:", len(self.dataset_train_classification))
        # print("Validate size:", len(self.dataset_validate_classification))
        # print("Test size:", len(self.dataset_test_classification))

        self.dl_train_classification = DataLoader(self.dataset_train_classification, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.dl_validate_classification = DataLoader(self.dataset_validate_classification, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        #self.dl_test_classification = DataLoader(self.dataset_test_classification, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

        print("Classification dataloaders created.")



    def get_detection_data_path(self):
        return self.detection_path
    
    def get_classification_data_path(self):
        return self.classification_path
    


    def load_inference_image(self, image_path):
        # transform = T.Compose([
        #     T.Resize((192, 272)),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # img = torchvision.io.decode_image(image_path, mode="RGB").to(torch.float32)/255.0
        return torchvision.io.decode_image(image_path, mode="RGB")#transform(img)

    



