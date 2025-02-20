#The main function is used to run the pipeline
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation

def main():
    dls = DataLoadingStation()
    mls = ModelLoadingStation()
    ts = TrainingStation(mls.model_detection, mls.model_classification, dls.dl_train_detection, dls.dl_validate_detection)
    ts.trainDetection(dls.dl_train_detection, dls.dl_validate_detection)

