import torch
import os
import json
import torchvision
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ClassifierTrainer import ClassifierTrainer
import AnalyticsModule as am
from torch.utils.data import DataLoader
from ultralytics.utils import DEFAULT_CFG

class TrainingStation:
    num_epochs_detection = 1
    num_epochs_classification = 3
    batch_size_detection = 2
    num_workers = 4
    

    #variables for YOLO training
    learning_rate_detection = 0.0001
    momentum_detection = 0.9
    weight_decay_detection = 0.0005
    frozen_layers_detection = 20


    model_detection = None
    model_classification = None
    
    yolo_trainer = None
    vit_trainer = None

    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)

    project_root = os.path.dirname(__file__)
    results_root = os.path.join(project_root, variables["results_root"])
    detection_results = os.path.join(results_root, variables["results_path_detection"])
    classification_results = os.path.join(results_root, variables["results_path_classification"])

    model_root = os.path.join(project_root, variables["model_root"])
    detection_model_path = os.path.join(model_root, variables["model_path_detection"])
    classification_model_path = os.path.join(model_root, variables["model_path_classification"])

    def __init__(self, model_detection : YOLO, model_classification):
        self.model_detection = model_detection
        self.model_classification = model_classification
        # if(model_detection != None):
        #     self.buildTrainer()# (data_loader_d_train).dataset , (data_loader_d_test).dataset)



    def buildTrainer(self):#, detection_dataset_train, detection_dataset_test):
        self.yolo_trainer = DetectionTrainer(overrides={
            'model' : self.model_detection.ckpt_path,
            'data' : 'data/detection/data.yaml',
            'epochs' : self.num_epochs_detection,
            'batch' : self.batch_size_detection,
            'save_dir' : self.detection_results,
            'workers' : self.num_workers,
            'imgsz' : 640,
            'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
            #'wdir' : self.detection_model_path,
            'save_period' : 0,#do not save checkpoints
            #'trainset' : detection_dataset_train,
            #'testset' : detection_dataset_test
        })
        # self.yolo_trainer.trainset = detection_dataset_train
        # self.yolo_trainer.testset = detection_dataset_test
        self.yolo_trainer.save_dir = (Path)(self.detection_results)
        self.yolo_trainer.wdir = (Path)(self.detection_model_path)


    
    def trainDetection(self):
        self.buildTrainer()
        self.yolo_trainer.train()
        #am.graph_detection(results)
        print("Detection training finished")

    def trainClassification(self, train_dl, valid_dl, test_dl, num_classes):
        self.vit_trainer = ClassifierTrainer(train_dl, valid_dl, test_dl, self.model_classification, num_classes)
        if(self.vit_trainer != None):
            self.vit_trainer.train()

    
    

