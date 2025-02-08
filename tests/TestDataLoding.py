import sys
import os
import unittest
import json

#adjusts the path to import the DataLoadingStation class - this assumes that the data class exists right outside the folder that contains these tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataLoadingStation import DataLoadingStation

class DataLoadingTests(unittest.TestCase):

    data_root = None
    dls = None
    variables = None

    detection_splits = {
        "train": 0.8,
        "validate": 0.1,
        "test": 0.1
    }

    classification_splits = {
        "train": 0.7,
        "validate": 0.1001,
        "test": 0.2
    }

    with open("variables.json", "r") as f:
        variables = json.load(f)
        data_root = variables["data_root"]

    def setUp(self):
        self.dls = DataLoadingStation()

    if __name__ == '__main__':
        unittest.main()

    
    #Test if the directories for both detection and classification datasets reachable
    #Asserts that the dataset directories exist and are not empty using the varaibles.json file
    def testDataDirectories(self):
        print("\nTesting data directories")
        print("Testing from path:", os.getcwd())
        print("Data root:", self.data_root)

        dataset_path_detection = self.dls.get_detection_data_path()
        print("Detection dataset path:", dataset_path_detection)
        self.assertTrue(os.path.isdir(dataset_path_detection))

        dataset_path_classification = self.dls.get_classification_data_path()
        print("Classification dataset path:", dataset_path_classification)
        self.assertTrue(os.path.isdir(dataset_path_classification))

        self.assertTrue(len(os.listdir(dataset_path_detection)) > 0)
        self.assertTrue(len(os.listdir(dataset_path_classification)) > 0)




    #Test the seed detection dataset loading. The structure of the dataset should be:
    #data/detection/-----------------------------------------------
    #                   |                   |                   |
    #                images/               labels/          data.yaml <- Not used in data loading, but may be needed for YOLO
    #                   |                   |
    #                  001...jpg            001...txt
    #                  002...jpg            002...txt
    #                  ...                   ...
    #This should load the dataset with the appropriate splits for training, validation, and testing.
    #Assert that the dataset is not empty and the sizes of splits are correct
    def testLoadDataDetection(self):
        print("\nTesting detection dataset loading")
        print("Splits are:", self.detection_splits)
        self.dls.load_data_detection(self.detection_splits["train"], self.detection_splits["validate"], self.detection_splits["test"])
        self.assertTrue(len(self.dls.dataset_train_detection) > 0)
        self.assertTrue(len(self.dls.dataset_validate_detection) > 0)
        self.assertTrue(len(self.dls.dataset_test_detection) > 0)

        self.assertTrue(len(self.dls.dataset_train_detection) == int(len(self.dls.dataset_detection) * self.dls.TRAIN_SPLIT_DETECTION))
        self.assertTrue(len(self.dls.dataset_validate_detection) == int(len(self.dls.dataset_detection) * self.dls.VALIDATE_SPLIT_DETECTION))
        self.assertTrue(len(self.dls.dataset_test_detection) == int(len(self.dls.dataset_detection) * self.dls.TEST_SPLIT_DETECTION))

        #asserts that combined lengths of splits are equal to the length of the dataset
        self.assertTrue(len(self.dls.dataset_train_detection) + len(self.dls.dataset_validate_detection) + len(self.dls.dataset_test_detection) == len(self.dls.dataset_detection))

    

    #Test the seed classification dataset loading. The structure of the dataset should be:
    #data/classification/
    #                   Achnatherum inebrians/ images 4002 - 4043
    #                   Achnatherum splendens/ images 3539 - 3581
    #                   ...                   ...
    #                   Zygophyllum xanthoxylon/ images 4194 - 4241
    # The labels for images are the names of folders that contain them.
    #This should load the dataset with the appropriate splits for training, validation, and testing.
    #Assert that the dataset is not empty and the sizes of splits are correct
    def testLoadDataClassification(self):
        print("\nTesting classification dataset loading")
        print("Splits are:", self.classification_splits)
        self.dls.load_data_classification(self.classification_splits["train"], self.classification_splits["validate"], self.classification_splits["test"])
        self.assertTrue(len(self.dls.dataset_train_classification) > 0)
        self.assertTrue(len(self.dls.dataset_validate_classification) > 0)
        self.assertTrue(len(self.dls.dataset_test_classification) > 0)

        self.assertTrue(len(self.dls.dataset_train_classification) == int(len(self.dls.dataset_classification) * self.dls.TRAIN_SPLIT_CLASSIFICATION))
        self.assertTrue(len(self.dls.dataset_validate_classification) == int(len(self.dls.dataset_classification) * self.dls.VALIDATE_SPLIT_CLASSIFICATION))
        self.assertTrue(len(self.dls.dataset_test_classification) == int(len(self.dls.dataset_classification) * self.dls.TEST_SPLIT_CLASSIFICATION))

        #asserts that combined lengths of splits are equal to the length of the dataset
        self.assertTrue(len(self.dls.dataset_train_classification) + len(self.dls.dataset_validate_classification) + len(self.dls.dataset_test_classification) == len(self.dls.dataset_classification))



