#The main function is used to run the pipeline
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation
from InferenceStation import InferenceStation

def main():
    print("Starting pipeline")
    print("Loading data")
    dls = DataLoadingStation()
    #dls.load_data_classification(0.9, 0.1001)# the test set wont be used so it does not matter what we set it to
    dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    mls = ModelLoadingStation()
    #mls.load_new_classification_model()
    mls.load_new_detection_standard_model()

    print("Training")
    #ts = TrainingStation(None, mls.cur_classification_model)
    ts = TrainingStation(mls.cur_detection_model, None)
    #ts.trainClassification(dls.dl_train_classification, dls.dl_validate_classification, dls.dl_test_classification, dls.num_classes)
    ts.trainDetection()
    print("Pipeline finished")

    # print("Starting inference pipeline")
    # print("looking for input image")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.9, 0.1001)# this is needed just to get the indexed classes
    # image = dls.load_inference_image("testInput.jpg")
    # mls = ModelLoadingStation()
    # mls.load_saved_classification_model()
    # mls.load_saved_detection_model()

    #print("Running inference")
    #inferS = InferenceStation()
    #print(inferS.inferOnClassification(image, mls.cur_classification_model, dls))
    #inferS.inferOnDetection("testInput.jpg", mls.cur_detection_model)
    #inferS.inferOnCombined("data/syntheticData/images/test/image5901588_0.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "testSyntheticOne.jpg")
    #inferS.inferOnSynthetic("data/syntheticData/data.yaml",mls.cur_detection_model, mls.cur_classification_model, dls, "testOutputSynthetic.jpg")


if __name__ == "__main__":
    main()