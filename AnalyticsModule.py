import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from InferenceStation import InferenceStation
from timeit import default_timer as timer 
# #t test with 10 trials, looking for confidence of 99%, t value is 2.821 - this is silly if we are testing on the validation dataset - the t-test can be used if we find a different dataset
# def t_test_classification(model, dls, inferS: InferenceStation):
#     results = []
#     for i in range(10):
#         results.append(inferS.inferOnClassificationAvg(model, dls))

#     for i in range(10):
#         print("Trial " + str(i) + ": " + str(results[i]))


def check_stats_classification(model, dls, inferS: InferenceStation, device, input_dtype = torch.float32):
    #print(model)
    start_time = timer()
    results = inferS.inferOnClassificationAvg(model, dls, device, input_dtype)
    end_time = timer()
    print("Inference time: " + str(end_time - start_time))
    
    print("Stat check: accuracy: {} precision: {} recall: {} f1: {}".format(results[0], results[1], results[2], results[3]))



def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    

def print_param_types(model):
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tensor.shape}, dtype: {tensor.dtype}")
        #else:
            #print(f"{name}: {tensor}") #this may cause an explosion
    
