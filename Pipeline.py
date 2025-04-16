#The main function is used to run the pipeline
import torch
from ultralytics import YOLO
from scripts.FileMover import find_and_move_file
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation
from InferenceStation import InferenceStation
from AnalyticsModule import check_stats_classification, print_size_of_model
from AnalyticsModule import check_stats_classification, check_stats_detection
from QuantStation import QuantStation

def main():

    # mls = ModelLoadingStation()
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.95, 0.05)
    # mls.load_saved_detection_model(3)
    # mls.load_saved_classification_model(2)
    # inferS = InferenceStation()
    # quantS = QuantStation(dls, inferS)

    # inferS.inferOnCombined("data/syntheticData/images/test/image5901588_5.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "output2.jpg")

    # print("Quantizing ViT")
    # quantizedViT = quantS.dynamic_quant_class(model = mls.cur_classification_model, dtype_incoming = torch.quint8)
    # print_size_of_model(quantizedViT)

    # inferS.inferOnCombined("data/syntheticData/images/test/image5901588_5.jpg", mls.cur_detection_model, quantizedViT, dls, "output1.jpg")
    


    # print("\nPipeline complete")
    pass


    

if __name__ == "__main__":
    main()