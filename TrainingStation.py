import torch
import os
import json
import torchvision
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import AnalyticsModule as am
from torch.utils.data import DataLoader
from ultralytics.utils import DEFAULT_CFG

class TrainingStation:
    num_epochs_detection = 1
    num_epochs_classification = 3
    batch_size_detection = 2
    batch_size_classification = 24
    num_workers = 4
    

    #variables for YOLO training
    learning_rate_detection = 0.0001
    momentum_detection = 0.9
    weight_decay_detection = 0.0005
    frozen_layers_detection = 20

    #variables for classification training
    learning_rate_classification = 0.0001
    num_heads = 2
    patch_size = 16


    model_detection = None
    model_classification = None
    
    yolo_trainer = None

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

    def __init__(self, model_detection, model_classification, data_loader_d, data_loader_c):
        self.model_detection = model_detection
        self.model_classification = model_classification
        self.buildTrainer( (data_loader_d).dataset , (data_loader_c).dataset)



    def buildTrainer(self, data_train, data_test):
        # self.yolo_trainer = BaseTrainer(
            # model=self.model_detection,
            # save_dir = self.detection_results,
            # wdir = self.detection_model_path,
            # save_period = 0, #disabled
            # batch_size = self.batch_size_detection,
            # epochs = self.num_epochs_detection,
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            # trainset = data_train,
            # testset = data_test
        # )
        self.yolo_trainer = BaseTrainer()
        self.yolo_trainer.model=self.model_detection,
        self.yolo_trainer.save_dir = self.detection_results,
        self.yolo_trainer.wdir = self.detection_model_path,
        self.yolo_trainer.save_period = 0, #disabled
        self.yolo_trainer.batch_size = self.batch_size_detection,
        self.yolo_trainer.epochs = self.num_epochs_detection,
        self.yolo_trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        self.yolo_trainer.trainset = data_train,
        self.yolo_trainer.testset = data_test



    
    def trainDetection(self, dl_train_detection, dl_validate_detection):
        results = (YOLO)(self.model_detection).train(trainer = self.yolo_trainer)
        am.graph_detection(results)
    
    

