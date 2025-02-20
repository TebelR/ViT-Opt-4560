#The main function is used to run the pipeline
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation



def main():
    print("Starting pipeline")
    print("Loading data")
    dls = DataLoadingStation()
    dls.load_data_detection(0.8, 0.1, 0.1)
    print("Loading model")
    mls = ModelLoadingStation()
    mls.load_new_detection_standard_model()
    print("Training")
    ts = TrainingStation(mls.cur_detection_model, None, dls.dl_train_detection, dls.dl_test_detection)
    ts.trainDetection(dls.dl_train_detection, dls.dl_validate_detection)
    print("Pipeline finished")


if __name__ == "__main__":
    main()