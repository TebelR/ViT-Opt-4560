#The main function is used to run the pipeline
import torch
from FileMover import find_and_move_file
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation
from InferenceStation import InferenceStation
from AnalyticsModule import check_stats_classification, check_stats_detection
from QuantStation import QuantStation

def main():
    print("Starting pipeline")
    # print("Loading data")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.9, 0.1001)# the test set wont be used so it does not matter what we set it to
    # dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_new_classification_model()
    # mls.load_new_detection_standard_model()

    #print("Training")
    #ts = TrainingStation(None, mls.cur_classification_model)
    #ts = TrainingStation(mls.cur_detection_model, None)
    #ts.trainClassification(dls.dl_train_classification, dls.dl_validate_classification, dls.dl_test_classification, dls.num_classes)
    #ts.trainDetection()
    #print("Pipeline finished")

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



    # print("Loading data")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.9, 0.1001)# the test set wont be used so it does not matter what we set it to
    # #dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_saved_classification_model()
    # #mls.load_saved_detection_model()

    # inferS = InferenceStation()
    # print("Analyzing performance")
    # #t test for classification and detection on n images of their respective datasets
    # check_stats_classification(mls.cur_classification_model, dls, inferS, "cuda")

    # print("Quantizing")
    # #quantize for various dtypes
    # quantS = QuantStation(dls, inferS)
    # quantS.set_class_model(mls.cur_classification_model)

    # # #quantize both dynamically and statically for int8
    # quantized_temp = quantS.dynamic_quant_class(mls.load_retrieve_saved_class_model(),dtype = torch.qint8)
    # print("Quantized for int8 dynamically...")
    # check_stats_classification(quantized_temp, dls, inferS, "cpu")

    # # quantized_temp = quantS.static_quant_class(mls.load_retrieve_saved_class_model(), dls)
    # # print("Quantized for int8 statically...")
    # # check_stats_classification(quantized_temp, dls, inferS, "cpu")

    # print("\nLoading Detection Data")
    # dls = DataLoadingStation()
    # dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_saved_detection_model()

    # print("\nBegin Training")
    # ts = TrainingStation(mls.cur_detection_model, None)
    # ts.trainDetection()

    # print("\nExporting")
    # export_dir = "models/detection/quantization"
    
    # print("EXPORTING FULL PRECISION")
    # # FP32 Export (Full Precision)
    # mls.cur_detection_model.export(format="engine")
    # logInfo = find_and_move_file("models/detection/", "models/detection/quantization", "yolo11n1.engine", "ModelFp32.engine")
    # print(logInfo)

    # print("EXPORTING HALF PRECISION")
    # # FP16 Export (Half Precision)
    # mls.cur_detection_model.export(format="engine", half=True)
    # logInfo = find_and_move_file("models/detection/", "models/detection/quantization", "yolo11n1.engine", "ModelFp16.engine")
    # print(logInfo)

    # print("EXPORTING QUANTIZED")
    # # INT8 Export (Quantized)
    # mls.cur_detection_model.export(format="engine", int8=True, data = "data/detection/data.yaml")
    # logInfo = find_and_move_file("models/detection/", "models/detection/quantization", "yolo11n1.engine", "ModelInt8.engine")
    # print(logInfo)

    print("\nCOMPARING METRICS")
    check_stats_detection("models/detection/quantization/ModelFp32.engine", "Fp32 Model")
    check_stats_detection("models/detection/quantization/ModelFp16.engine", "Fp16 Model")
    check_stats_detection("models/detection/quantization/ModelInt8.engine", "Int8 Model")


    print("\nPipeline complete")



    

if __name__ == "__main__":
    main()