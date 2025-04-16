#The main function is used to run the pipeline
import torch
from FileMover import find_and_move_file
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation
from InferenceStation import InferenceStation
from AnalyticsModule import check_stats_classification, print_size_of_model, check_stats_detection
from QuantStation import QuantStation

def main():
    pass




    

if __name__ == "__main__":
    main()