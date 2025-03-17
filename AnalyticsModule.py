import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from InferenceStation import InferenceStation

# #t test with 10 trials, looking for confidence of 99%, t value is 2.821 - this is silly if we are testing on the validation dataset - the t-test can be used if we find a different dataset
# def t_test_classification(model, dls, inferS: InferenceStation):
#     results = []
#     for i in range(10):
#         results.append(inferS.inferOnClassificationAvg(model, dls))

#     for i in range(10):
#         print("Trial " + str(i) + ": " + str(results[i]))


def check_stats_classification(model, dls, inferS: InferenceStation, device, input_dtype = torch.float32):
    #print(model)
    results = inferS.inferOnClassificationAvg(model, dls, device, input_dtype)
    
    print("Stat check: accuracy: {} precision: {} recall: {} f1: {}".format(results[0], results[1], results[2], results[3]))
    
def check_stats_detection(model, name):
    inferS = InferenceStation()
    print("\n" + name)
    metricsInfo = inferS.inferQuantizedDetection(model)
    # print("Stat check: f1: {} mAP50: {} mAP50-95: {} Recall: {} Latency (ms): {} FPS: {} Model Size (MB): {} Total Time (s): {}".format(metricsInfo[1], metricsInfo[2], metricsInfo[3], metricsInfo[4], metricsInfo[5], metricsInfo[6], metricsInfo[7], metricsInfo[8]))
    print("\n---------Evaluation Results---------")
    for key, value in metricsInfo.items():
        print(f"{key}: {value}")