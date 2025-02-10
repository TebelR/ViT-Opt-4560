
#Tests for the class that handles loading and saving models. Models in here are mocked because they take a long time to load.
#These tests should not mess up the current best indices in variables.json
from ModelLoadingStation import ModelLoadingStation
import os
import json
import torch
import pandas as pd
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock



class TestModelLoading(unittest.TestCase):

    variables = None
    with open("variables.json", "r") as f:
        variables = json.load(f)

    project_root = os.path.join(os.path.dirname(__file__), "..")
    model_root = os.path.join(project_root, variables["model_root"])
    detection_model_path = os.path.join(model_root, variables["model_path_detection"])
    classification_model_path = os.path.join(model_root, variables["model_path_classification"])

    def setUp(self):
        pass


    #Test instantiating the ModelLoadingStation class
    #Assert that it is not null
    def test_1_ModelLoadingStation(self):
        print("\nTesting the instantiation of ModelLoadingStation")
        mls = ModelLoadingStation()
        self.assertIsNotNone(mls)




    #Test loading brand new detection and classification models
    #Assert that they are not null
    def test_2_LoadNewModels(self):
        print("\nTesting the loading of new models")
        mls = ModelLoadingStation()
        mls.load_new_detection_standard_model()
        mls.load_new_classification_model()
        self.assertIsNotNone(mls.cur_detection_model)
        self.assertIsNotNone(mls.cur_classification_model)

    
    #Test saving the models to an index of 0
    #Assert that they appear in the model directory
    def test_3_SaveModels(self):
        print("\nTesting the saving of models")
        mls = ModelLoadingStation()
        mls.load_new_detection_standard_model()
        mls.load_new_classification_model()
        mls.overwrite_saved_classification_model(0)
        mls.overwrite_saved_detection_model(0)
        self.assertTrue(os.path.exists(os.path.join(self.detection_model_path, "yolo11n0.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.classification_model_path, "vit0.pt")))

    #Test that the index is incremented when saving a model
    #Assert that the index is 1 after saving a model, this will set the index back to what it was after the test
    def test_4_IncrementIndex(self):
        print("\nTesting the increment of the index")
        mls = ModelLoadingStation()
        best_det_index = mls.best_detection_index
        print("Best detection index: " + str(mls.best_detection_index))

        mls.best_detection_index = -1
        mls.load_new_detection_standard_model()
        mls.save_detection_model()
        print("Closing down model loading station")
        #close down the old class, making it write to variables.json
        del mls
        with open("variables.json", "r") as f:
            self.variables = json.load(f)#refresh the variables
            print("Refreshed variables")
        
        #check that the index increased
        self.assertTrue(self.variables["best_detection_index"] == 0)
        mls = ModelLoadingStation()
        print("Loaded new model loading station, changing index")
        mls.best_detection_index = best_det_index
        del mls#close down the new class, making it write to variables.json and restoring the index


    #Test loading a saved model
    #Assert that the model is not null after loading
    #This assumes that some models exist in the model directory
    def test_5_LoadSavedModel(self):
        print("\nTesting the loading of saved models")
        mls = ModelLoadingStation()
        mls.load_saved_detection_model()
        mls.load_saved_classification_model()
        self.assertIsNotNone(mls.cur_detection_model)
        self.assertIsNotNone(mls.cur_classification_model)

        


    