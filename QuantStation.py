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
import torch.ao.quantization.qconfig as qconfig
from DataLoadingStation import DataLoadingStation
import torchvision.transforms as T

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
        
        config_dict = {
            "features.0": None,
            "features.0.0": qconfig.per_channel_dynamic_qconfig,#config for the Conv2d module

            "features.1": None,#this is a swin transformer block
            #"features.1.0.attn.proj": qconfig.default_qconfig,#config for one of the Linear modules in the first Swin block
            "features.1.0.mlp.0": qconfig.default_qconfig,#linear
            "features.1.0.mlp.3": qconfig.default_qconfig,#linear

            #"features.1.1.atn.proj": qconfig.default_qconfig,#linear attention
            "features.1.1.mlp.0": qconfig.default_qconfig,#linear
            "features.1.1.mlp.3": qconfig.default_qconfig,#linear

            "features.2": None,# this is a PatchMergingV2 - no touch

            "features.3": None,#this is a swin transformer block
            #"features.3.0.attn.proj": qconfig.default_qconfig,#config for one of the Linear modules in the first Swin block
            "features.3.0.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.3.0.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.3.0.mlp.0": qconfig.default_qconfig,#linear
            "features.3.0.mlp.3": qconfig.default_qconfig,#linear

           #"features.3.1.atn.proj": qconfig.default_qconfig,#linear attention
            "features.3.1.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.3.1.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.3.1.mlp.0": qconfig.default_qconfig,#linear
            "features.3.1.mlp.3": qconfig.default_qconfig,#linear

            "features.4": None,#this is a PatchMergingV2 - no touch

            "features.5": None,#this one contains many transformer blocks
            #"features.5.0.attn.proj": qconfig.default_qconfig,#linear
            "features.5.0.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.5.0.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.5.0.mlp.0": qconfig.default_qconfig,#linear
            "features.5.0.mlp.3": qconfig.default_qconfig,#linear

            #"features.5.1.attn.proj": qconfig.default_qconfig,#linear
            "features.5.1.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.5.1.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.5.1.mlp.0": qconfig.default_qconfig,#linear
            "features.5.1.mlp.3": qconfig.default_qconfig,#linear

            #"features.5.2.attn.proj": qconfig.default_qconfig,#linear
            "features.5.2.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.5.2.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.5.2.mlp.0": qconfig.default_qconfig,#linear
            "features.5.2.mlp.3": qconfig.default_qconfig,#linear

            #"features.5.3.attn.proj": qconfig.default_qconfig,#linear
            "features.5.3.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.5.3.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.5.3.mlp.0": qconfig.default_qconfig,#linear
            "features.5.3.mlp.3": qconfig.default_qconfig,#linear

            #"features.5.4.attn.proj": qconfig.default_qconfig,#linear
            "features.5.4.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.5.4.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.5.4.mlp.0": qconfig.default_qconfig,#linear
            "features.5.4.mlp.3": qconfig.default_qconfig,#linear

            #"features.5.5.attn.proj": qconfig.default_qconfig,#linear
            "features.5.5.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU            
            "features.5.5.attn.cpb_mlp.2": qconfig.default_qconfig,#linear            
            "features.5.5.mlp.0": qconfig.default_qconfig,#linear
            "features.5.5.mlp.3": qconfig.default_qconfig,#linear

            "features.6": None,#this is a PatchMergingV2

            "features.7": None,#this has two transformer blocks
            #"features.7.0.attn.proj": qconfig.default_qconfig,#linear
            "features.7.0.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.7.0.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.7.0.mlp.0": qconfig.default_qconfig,#linear
            "features.7.0.mlp.3": qconfig.default_qconfig,#linear

            #"features.7.1.attn.proj": qconfig.default_qconfig,#linear
            "features.7.1.attn.cpb_mlp.0": qconfig.default_qconfig,#linear - index 1 is a ReLU
            "features.7.1.attn.cpb_mlp.2": qconfig.default_qconfig,#linear
            "features.7.1.mlp.0": qconfig.default_qconfig,#linear
            "features.7.1.mlp.3": qconfig.default_qconfig,#linear

            "norm": None,
            "avgpool": None,
            "flatten": None,
            "head": qconfig.default_qconfig,# this is a linear too
        }

        quant_model = torch.quantization.quantize_dynamic(model, config_dict, dtype=dtype)
        return quant_model
    
    def static_quant_class(self, model = None, dls : DataLoadingStation = None):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")#move the model to the cpu as it may misbehave on the gpu
        model.eval()
        torch.backends.quantized.engine = 'x86'
        config = get_default_qconfig_mapping('x86')# the default qconfig for pc is x86

        #config.set_global(qconfig.float16_static_qconfig)#This will make it so that things quantize to float16

        # config.set_object_type(torch.nn.Conv2d, qconfig.default_qconfig)
        # config.set_object_type(torch.nn.Linear, qconfig.float_qparams_weight_only_qconfig)#quantization with Pytorch's FX fuses linear layers and their activations
        # config.set_object_type(torch.nn.modules.activation.ReLU, qconfig.float_qparams_weight_only_qconfig)#ReLU needs to be quantized the same way as its linear layer
        # config.set_object_type(torch.nn.modules.activation.GELU, None)#omit the GELU - might be able to replace this with a swish later
        # config.set_object_type(torch.nn.LayerNorm, None)#omit the layer norm as this breaks the ViT
        # config.set_object_type(torch.nn.Softmax, None)

        config.set_module_name("features[0]", None)
        config.set_module_name("features[1]", None)
        config.set_module_name("features[2]", None)
        config.set_module_name("features[3]", None)
        config.set_module_name("features[4]", None)
        config.set_module_name("features[5]", None)
        config.set_module_name("features[6]", None)
        config.set_module_name("features[7]", None)

        config.set_module_name("norm", None)
        config.set_module_name("avgpool", None)
        config.set_module_name("flatten", None)
        config.set_module_name("head", None)

        num_calibration_imgs = 100
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#calibration data needs to look the same as what the model would typically see
        ])
        images = []
        for i in range(num_calibration_imgs):
            random_choice = random.randint(0, len(dls.dataset_classification) - 1)
            image, _ = dls.dataset_classification[random_choice]
            images.append(transform(image))

        
        images = torch.stack(images).to("cpu")

        prepared_model = prepare_fx(model, config, example_inputs=images)

        #calibration - so that the observers can see the range of values
        with torch.no_grad():
            prepared_model(images)

        quantized_model = convert_fx(prepared_model)
        return  quantized_model



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

    