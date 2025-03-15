import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.quantization
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping
from DataLoadingStation import DataLoadingStation
from InferenceStation import InferenceStation

class QuantStation():

    orig_class_model = None
    orig_detect_model = None
    dls = None
    inferS = None

    project_root = os.path.dirname(__file__)
    save_dir = os.path.join(project_root, "quantization")
    
    def __init__(self, dls : DataLoadingStation, inferS : InferenceStation):
        self.dls = dls
        self.inferS = inferS
        os.makedirs(self.save_dir, exist_ok=True)
        #print(torch.backends.quantized.engine)# the default is x86
        #torch.backends.quantized.engine = 'fbgemm'
        # print(torch.backends.quantized.engine)


    def set_class_model(self, model):
        self.orig_class_model = model

    def set_detect_model(self, model):
        self.orig_detect_model = model


    def dynamic_quant_class(self, model = None, dtype = torch.qint8):
        if model == None:
            print("No model passed into the dynamic quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")
        model.eval()
        
        quant_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)
        #self.convert_class_biases(quant_model, dtype)
        return model
    
    def static_quant_class(self, model = None):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")
        model.eval()
        config = get_default_qconfig_mapping('x86')# this is the config for the default engine on a pc with a x86 cpu
        calibration_input = torch.randn(1, 3, 192, 272, dtype=torch.float32)
        calibration_input = calibration_input.to(memory_format=torch.channels_last)

        prepared_model = prepare_fx(model, config, example_inputs=calibration_input)

        #calibration - so that the observers can see the range of values
        with torch.no_grad():
            prepared_model(calibration_input)

        quantized_model = convert_fx(prepared_model)
        return quantized_model




#this is a replacement for the GELU function in the Swin ViT
#This is much better to quantize than GELU
class Swish(torch.nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
    

#The ViT contains a layer norm that is not quantizable - need to find a way to replace it with something that is similar in function
# class FusedLinear(nn.Module.LayerNorm):
#     def forward(self, x, dtype = torch.quint8):
#         return super().forward(x.float()).to(x.dtype)




#-------------------------------------------------------------------------YOLO methods-------------------------------------------------------------------------

    