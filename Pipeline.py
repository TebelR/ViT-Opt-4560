#The main function is used to run the pipeline
import torch
from FileMover import find_and_move_file
from DataLoadingStation import DataLoadingStation
from TrainingStation import TrainingStation
from ModelLoadingStation import ModelLoadingStation
from InferenceStation import InferenceStation
from AnalyticsModule import check_stats_classification, print_size_of_model,print_param_types
from AnalyticsModule import check_stats_classification, check_stats_detection
from QuantStation import QuantStation

def main():
    print("Starting pipeline")
    # print("Loading data")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.95, 0.05)# the test set wont be used so it does not matter what we set it to
    # #dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_new_classification_model()
    # #mls.load_new_detection_standard_model()

    # print("Training")
    # ts = TrainingStation(None, mls.cur_classification_model)
    # #ts = TrainingStation(mls.cur_detection_model, None)
    # ts.trainClassification(dls.dl_train_classification, dls.dl_validate_classification, dls.dl_test_classification, dls.num_classes)
    # #ts.trainDetection()
    # print("Pipeline finished")

    print("Starting inference pipeline")
    print("looking for input image")
    dls = DataLoadingStation()
    dls.load_data_classification(0.95, 0.05)# this is needed just to get the indexed classes
    #image = dls.load_inference_image("testInput.jpg")
    mls = ModelLoadingStation()
    mls.load_saved_classification_model(2)
    mls.load_saved_detection_model(5) #2 is 2500x2500, 3 is 1600x1600, 4 is 1280x1280 and 5 is 960x960

    print("Running inference")
    inferS = InferenceStation()
    #print(inferS.inferOnClassification(image, mls.cur_classification_model, dls))
    #inferS.inferOnDetection("testInput.jpg", mls.cur_detection_model)
    #inferS.inferOnCombined("data/syntheticForTest/images/test/image6065345_0.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "testForLabels1.jpg")
    #inferS.inferOnCombined("data/syntheticForTest/images/test/image6065345_1.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "testForLabels2.jpg")
    inferS.inferOnCombined("data/syntheticForTest/images/test/image6065345_2.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "testForLabels3.jpg")
    #inferS.inferOnCombined("data/syntheticForTest/images/test/image6065345_3.jpg", mls.cur_detection_model, mls.cur_classification_model, dls, "testForLabels4.jpg")
    #inferS.inferOnSynthetic("data/syntheticData/data.yaml",mls.cur_detection_model, mls.cur_classification_model, dls, "testOutputSynthetic.jpg")



    # print("Loading data")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.9, 0.1001)# the test set wont be used so it does not matter what we set it to
    # #dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_saved_classification_model()
    # #mls.load_saved_detection_model()

    #inferS.evaluateCombined("data/syntheticData", mls.cur_detection_model, mls.cur_classification_model, dls)


    # print("Loading data")
    # dls = DataLoadingStation()
    # dls.load_data_classification(0.9, 0.1001)# the test set wont be used so it does not matter what we set it to
    # #dls.load_data_detection(0.8, 0.05, 0.15)# the test set wont be used so it does not matter what we set it to
    # mls = ModelLoadingStation()
    # mls.load_saved_classification_model()
    # #mls.load_saved_detection_model()


    #print("Analyzing performance on CPU")
    #check_stats_classification(mls.cur_classification_model, dls, inferS, "cpu")

    #print("Analyzing performance on CUDA")
    #check_stats_classification(mls.cur_classification_model, dls, inferS, "cuda")

    # print("Quantizing")
    # # #quantize for various dtypes
    # quantS = QuantStation(dls, inferS)
    # quantS.set_class_model(mls.cur_classification_model)

    # print_size_of_model(mls.cur_classification_model)

    # #quantize both dynamically and statically for int8
    # quantized_temp = quantS.dynamic_quant_class(torch.qint8, mls.load_retrieve_saved_class_model())
    # print_size_of_model(quantized_temp)
    # check_stats_classification(quantized_temp, dls, inferS, "cpu")
    

    

    # quantized_int8 = quantS.dynamic_quant_class(torch.qint8,model = mls.load_retrieve_saved_class_model())
    # print_size_of_model(quantized_int8)
    # check_stats_classification(quantized_int8, dls, inferS, "cpu")
    
    # quantized_int8 = mls.load_retrieve_saved_class_model()
    # print_size_of_model(quantized_int8)
    # check_stats_classification(quantized_int8, dls, inferS, "cpu")
    

    # quantized_f16 = mls.load_retrieve_saved_class_model()
    # quantS.dynamic_quant_class(torch.float16, quantized_f16)
    # print_size_of_model(quantized_f16)
    # check_stats_classification(quantized_f16, dls, inferS, "cuda")


    # quantS.static_quant_class(model = quantized_f16, dls = dls,  dtype_incoming = torch.float16)
    # print_size_of_model(quantized_f16)
    # check_stats_classification(quantized_f16, dls, inferS, "cuda")

    


    #quantized_f16 = quantS.dynamic_quant_class(torch.float16, quantized_f16)
    

    # quantized_temp = quantS.dynamic_quant_class(torch.float16,mls.load_retrieve_saved_class_model())
    # print_size_of_model(quantized_temp)
    # check_stats_classification(quantized_temp, dls, inferS, "cpu")
    #print_size_of_model(quantized_temp)

    #quantized_temp = quantS.static_quant_class(mls.load_retrieve_saved_class_model(), dls)
    #quantized_temp = quantS.static_graph_quant_class(mls.load_retrieve_saved_class_model(), dls) - this results in near 0 accuracy
    #print("Quantized for int8 statically...")
    #check_stats_classification(quantized_temp, dls, inferS, "cpu")
    
    #print_size_of_model(quantized_temp)

    #print_param_types(quantized_temp)

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

    # print("\nCOMPARING METRICS")
    # check_stats_detection("models/detection/quantization/ModelFp32.engine", "Fp32 Model")
    # check_stats_detection("models/detection/quantization/ModelFp16.engine", "Fp16 Model")
    # check_stats_detection("models/detection/quantization/ModelInt8.engine", "Int8 Model")


    print("\nPipeline complete")



    

if __name__ == "__main__":
    main()