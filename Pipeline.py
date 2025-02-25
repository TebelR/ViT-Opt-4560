#The main function is used to run the pipeline
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation



def main():
    print("Starting pipeline")
    print("Loading data")
    dls = DataLoadingStation()
    dls.load_data_classification(0.7, 0.1001, 0.2)
    mls = ModelLoadingStation()
    mls.load_new_classification_model()

    print("Training")
    ts = TrainingStation(None, mls.cur_classification_model)
    ts.trainClassification(dls.dl_train_classification, dls.dl_validate_classification, dls.dl_test_classification, dls.num_classes)
    print("Pipeline finished")


if __name__ == "__main__":
    main()