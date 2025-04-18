#This station is reponsible for loading/saving models or creating new ones before training and testing.
#The pipeline comes to this class before training and right after training.

#The original models for our tasks are as follows:
#Detection: YOLOv11 nano from ultralytics - pretrained on COCO - to be fine-tuned on our dataset. The normal YOLOv11 is used instead of obb because of the dataset.
#Classification: SwinViT_t v2 from PyTorch - pretrained on ImageNet - to be fine-tuned on our dataset


import torch
import os
import json
import torchvision
from ultralytics import YOLO
from torch import nn

class ModelLoadingStation:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)
    best_detection_index = variables["best_detection_index"]
    best_classification_index = variables["best_classification_index"]

    project_root = os.path.dirname(__file__)
    model_root = os.path.join(project_root, variables["model_root"])
    detection_model_path = os.path.join(model_root, variables["model_path_detection"])
    classification_model_path = os.path.join(model_root, variables["model_path_classification"])

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
        print("Saved best model indices to variables.json")


    #Downloads a pretrained YOLOv11 model from ultralytics and saves it to the model directory.
    def load_new_detection_standard_model(self):
        self.cur_detection_model = YOLO(os.path.join(self.detection_model_path, "yolo11n.pt"))
        print("Loaded a new detection model.")

    #Loads a saved detection model from the model directory based on index. If no index is given - this loads the model with the highest index (newest one).
    def load_saved_detection_model(self,index=None):
        if index is None:
            self.cur_detection_model = YOLO(os.path.join(self.detection_model_path, "yolo11n" + str(self.best_detection_index) + ".pt"))
            print("Loaded a saved detection model. Index: " + str(self.best_detection_index))
        else:
            self.cur_detection_model = YOLO(os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt"))
            print("Loaded a saved detection model. Index: " + str(index))

    #Saves the current detection model and overwirtes the specified index with it.
    def overwrite_saved_detection_model(self,index):
        if os.path.exists(os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt")):
            save_path = os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt")
            self.cur_detection_model.save(save_path)
            print("Overwrote a detection model. Index: " + str(index))
        else:
            save_path = os.path.join(self.detection_model_path, "yolo11n" + str(index) + ".pt")
            self.cur_detection_model.save(save_path)
            print("Tried to overwrite a detection model that did not exist. Saving it instead at index: " + str(index))


    #Saves the current detection model to the model directory and appends the index to the filename.
    def save_detection_model(self):
        old_index = self.best_detection_index
        try:
            self.best_detection_index += 1
            save_path = os.path.join(self.detection_model_path, "yolo11n" + str(self.best_detection_index) + ".pt")
            self.cur_detection_model.save(save_path)
            print("Saved a detection model. Index: " + str(self.best_detection_index))
        except Exception as e:
            self.best_detection_index = old_index
            print("Failed to save detection model. Index: " + str(self.best_detection_index))
            print(e)


    #Classification section----------------------------------------------------------------------------------------------------------------------------

    #Loads a saved classification model from the model directory based on index. If no index is given - this loads the model with the highest index (newest one).
    #The model itself exists in models/vit-model.pt
    #This will load the model and then look for best weights stored in models/classification/vit#.pt

    def load_retrieve_saved_class_model(self, num_classes = 88):
        output = torch.load(os.path.join(self.model_root, "vit-model.pt"), weights_only=False)
        in_features = output.head.in_features
        output.head = nn.Linear(in_features, num_classes)
        output.load_state_dict(torch.load(os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt")))
        return output

    def load_saved_classification_model(self,index=None, num_classes = 88):
        if index is None:
            # self.cur_classification_model = torch.load(os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt"))
            self.cur_classification_model = torch.load(os.path.join(self.model_root, "vit-model.pt"), weights_only=False)
            in_features = self.cur_classification_model.head.in_features
            self.cur_classification_model.head = nn.Linear(in_features, num_classes)
            self.cur_classification_model.load_state_dict(torch.load(os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt")))
            self.cur_classification_model.to(self.device)
            print("Loaded a saved classification model. Index: " + str(self.best_classification_index))
        else:
            self.cur_classification_model = torch.load(os.path.join(self.model_root, "vit-model.pt"), weights_only=False)
            in_features = self.cur_classification_model.head.in_features
            self.cur_classification_model.head = nn.Linear(in_features, num_classes)
            self.cur_classification_model.load_state_dict(torch.load(os.path.join(self.classification_model_path, "vit" + str(index) + ".pt")))
            print("Loaded a saved classification model. Index: " + str(index))
        

    
    def load_new_classification_model(self):
        self.cur_classification_model = torchvision.models.swin_v2_t(weights='DEFAULT').to(self.device)
        print("Loaded a new classification model.")
        

    #Saves the current classification model and overwirtes the specified index with it.
    def overwrite_saved_classification_model(self,index):
        if os.path.exists(os.path.join(self.classification_model_path, "vit" + str(index) + ".pt")):
            save_path = os.path.join(self.classification_model_path, "vit" + str(index) + ".pt")
            torch.save(self.cur_classification_model.state_dict(), save_path)
            print("Overwrote a classification model. Index: " + str(index))
        else:
            save_path = os.path.join(self.classification_model_path, "vit" + str(index) + ".pt")
            torch.save(self.cur_classification_model.state_dict(), save_path)
            print("Tried to overwrite a classification model that did not exist. Saving it instead at index: " + str(index))

    #Saves the current classification model to the model directory and appends the index to the filename.
    def save_classification_model(self):
        old_index = self.best_classification_index
        try:
            self.best_classification_index += 1
            save_path = os.path.join(self.classification_model_path, "vit" + str(self.best_classification_index) + ".pt")
            torch.save(self.cur_classification_model.state_dict(), save_path)
            print("Saved a classification model. Index: " + str(self.best_classification_index))
        except Exception as e:
            self.best_classification_index = old_index
            print("Failed to save classification model. Index: " + str(self.best_classification_index))
            print(e)


    def load_trained_classification_model(self):
        self.cur_classification_model = torch.load(os.path.join(self.model_root, "vit-model.pt"), weights_only=False)
        self.cur_classification_model.load_state_dict(torch.load(os.path.join(self.project_root, "runs/classify/best/weights/best.pt")))
        self.cur_classification_model.to(self.device)
        print("Loaded a trained classification model.")