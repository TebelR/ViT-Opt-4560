#This station is reponsible for loading/saving models or creating new ones before training and testing.
#The pipeline comes to this class before training and right after training.

#The original models for our tasks are as follows:
#Detection: YOLOv11 nano from ultralytics - pretrained on COCO and fine-tuned on our dataset. The normal YOLOv11 is used instead of obb because of the dataset.
#Classification: ViT from PyTorch - pretrained on ImageNet and fine-tuned on our dataset


import torch
import os
import json
import torchvision
from ultralytics import YOLO

class ModelLoadingStation:
    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)
    best_detection_index = variables["best_detection_index"]
    best_classification_index = variables["best_classification_index"]

    project_root = os.path.dirname(__file__)
    model_root = os.path.join(project_root, variables["model_root"])
    detection_model_path = os.path.join(model_root, variables["detection_model_path"])
    classification_model_path = os.path.join(model_root, variables["classification_model_path"])

    cur_detection_model = None
    cur_classification_model = None

    def __init__(self):
        pass

    #Destructor method that saves the best indices to the variables.json file.
    def __del__(self):
        with open("variables.json", "w") as f:
            self.variables["best_detection_index"] = self.best_detection_index
            self.variables["best_classification_index"] = self.best_classification_index
            json.dump(self.variables, f)


    #Downloads a pretrained YOLOv11 model from ultralytics and saves it to the model directory.
    def load_new_detection_standard_model(self):
        self.cur_detection_model = YOLO("yolo11n.pt")

    #Loads a saved detection model from the model directory based on index. If no index is given - this loads the model with the highest index (newest one).
    def load_saved_detection_model(self,index=None):
        if index is None:
            self.cur_detection_model = YOLO(os.path.join(self.detection_model_path, "yolo11n" + str(self.best_detection_index) + ".pt"))
        else:
            self.cur_detection_model = YOLO(os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt"))

    #Saves the current detection model and overwirtes the specified index with it.
    def overwrite_saved_detection_model(self,index):
        os.remove(os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt"))
        torch.save(self.cur_detection_model, os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt"))


    #Saves the current detection model to the model directory and appends the index to the filename.
    def save_detection_model(self):
        self.best_detection_index += 1
        torch.save(self.cur_detection_model, os.path.join(self.detection_model_path, "yolo11n" + str(self.best_detection_index) + ".pt"))






    #Classification section----------------------------------------------------------------------------------------------------------------------------

    #Loads a saved classification model from the model directory based on index. If no index is given - this loads the model with the highest index (newest one).
    def load_saved_classification_model(self,index=None):
        if index is None:
            self.cur_classification_model = torch.load(os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt"))
        else:
            self.cur_classification_model = torch.load(os.path.join(self.classification_model_path, "vit" + str(index) + ".pt"))


    
    def load_new_classification_model(self):
        self.cur_classification_model = torchvision.models.swin_v2_t(pretrained=True,weights='DEFAULT')


    #Saves the current classification model and overwirtes the specified index with it.
    def overwrite_saved_classification_model(self,index):
        os.remove(os.path.join(self.classification_model_path, "vit" + str(index) + ".pt"))
        torch.save(self.cur_classification_model, os.path.join(self.classification_model_path, "vit" + str(index) + ".pt"))


    #Saves the current classification model to the model directory and appends the index to the filename.
    def save_classification_model(self):
        self.best_classification_index += 1
        torch.save(self.cur_classification_model, os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt"))

    



    







