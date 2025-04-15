import os
import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
from InferenceStation import InferenceStation
from timeit import default_timer as timer 




#Takes in an inference station instance and a classification model and checks the stats of the model. Returns nothing
def check_stats_classification(model, dls, inferS: InferenceStation, device, input_dtype = torch.float32):
    #print(model)
    start_time = timer()
    results = inferS.inferOnClassificationAvg(model, dls, device, input_dtype)
    end_time = timer()
    print("Inference time: " + str(end_time - start_time))
    
    print("Stat check: accuracy: {} precision: {} recall: {} f1: {}".format(results[0], results[1], results[2], results[3]))



#Prints the size of the given model and counts the number of float and int parameters in it. Returns nothing.
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

    num_floats = 0
    num_ints = 0
    for name,tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
                num_floats += tensor.numel()
            else:
                num_ints += tensor.numel()
    print(f"Number of floats: {num_floats}")
    print(f"Number of ints: {num_ints}")
    #print_param_types(model)

# def print_param_types(model):
#     for name, tensor in model.state_dict().items():
#         if isinstance(tensor, torch.Tensor):
#             print(f"{name}: {tensor.shape}, dtype: {tensor.dtype}")
#         #else:
#             #print(f"{name}: {tensor}") #this may cause an explosion
    


#Similar to check_stats_classification but for object detection with YOLO from ultralytics
def check_stats_detection(model, name):
    inferS = InferenceStation()
    print("\n" + name)
    metricsInfo = inferS.inferQuantizedDetection(model)
    # print("Stat check: f1: {} mAP50: {} mAP50-95: {} Recall: {} Latency (ms): {} FPS: {} Model Size (MB): {} Total Time (s): {}".format(metricsInfo[1], metricsInfo[2], metricsInfo[3], metricsInfo[4], metricsInfo[5], metricsInfo[6], metricsInfo[7], metricsInfo[8]))
    print("\n---------Evaluation Results---------")
    for key, value in metricsInfo.items():
        print(f"{key}: {value}")