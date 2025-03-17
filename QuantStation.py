import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.ao.quantization as quant
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping, prepare, convert, HistogramObserver, MinMaxObserver, QConfig, MovingAverageMinMaxObserver, default_weight_observer
import torch.fx as fx
import torch.ao.quantization.qconfig_mapping as qc_mapping
from DataLoadingStation import DataLoadingStation
from InferenceStation import InferenceStation
import torch.ao.quantization.qconfig as qconfig
from DataLoadingStation import DataLoadingStation
import torchvision.transforms as T
from torchvision.models.swin_transformer import SwinTransformerBlockV2

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
            "avgpool": qconfig.default_qconfig, #avg pool can be quantized
            "flatten": None,
            "head": qconfig.default_qconfig,# this is a linear too
        }

        quant_model = torch.quantization.quantize_dynamic(model, config_dict, dtype=dtype)
        return quant_model
    


    def setup_static_classification_qconfig(self, model):
        activation_observer = HistogramObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, bins=256)
        weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-1.0, quant_max=1.0)

        linear_custom_qconfig = QConfig(activation = activation_observer, weight = weight_observer)


        #self.qconfig_application_rec(model, linear_custom_qconfig)
        model.qconfig = qconfig.default_qconfig
        print(model.qconfig)
        

    def qconfig_application_rec(self, model, linear_custom_qconfig):
        for name, module in model.named_children():
            
            if isinstance(module, torch.nn.Conv2d):
                # module.qconfig = QConfig(
                # activation=MovingAverageMinMaxObserver.with_args(dtype = torch.qint8,qscheme=torch.per_tensor_affine),#if this throws an error, try per_tensor_symmetric
                # weight=default_weight_observer
                # )
                module.qconfig = None
            elif isinstance(module, torch.nn.Linear):
                module.qconfig = qconfig.get_default_qconfig("onednn")#linear_custom_qconfig
            else:
                module.qconfig = None #do not touch anything else
            self.qconfig_application_rec(module, linear_custom_qconfig)#apply this recursively to submodules
            
        

    
    def static_quant_class(self, model = None, dls : DataLoadingStation = None):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")#move the model to the cpu as it may misbehave on the gpu
        model.eval()

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

        #define a custom qconfig for linear modules
        linear_custom_qconfig = QConfig(
            activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, bins=96, reduce_range = True),
            weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
        
        #also a custom qconfig for Conv2D modules
        conv2d_custom_qconfig = QConfig(
            activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, bins=64, reduce_range = True),
            weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )

        custom_qconfig_mapping = qc_mapping.QConfigMapping()
        custom_qconfig_mapping.set_global(None)#ignore everything except for certain modules defined below

        for name, module in model.features.named_modules():#fuse the linear and ReLU for better performance
            if isinstance(module, SwinTransformerBlockV2):
                quant.fuse_modules(module, ["attn.cpb_mlp.0", "attn.cpb_mlp.1"], inplace=True)

        for name, module in model.named_modules():#then apply custom qconfigs to selected modules - this is in a separate loop because otherwise there's a recursion overflow
            if isinstance(module, torch.nn.Conv2d):
                custom_qconfig_mapping.set_module_name(name, conv2d_custom_qconfig)

            elif isinstance(module, torch.nn.Linear):# and "qkv" not in name:# and "head" not in name:#DO not touch qkv and head linear layers
                custom_qconfig_mapping.set_module_name(name, linear_custom_qconfig)


        torch.backends.quantized.engine = "fbgemm"
        prepared_model = prepare_fx(model, custom_qconfig_mapping, example_inputs=images)

        # for name, module in prepared_model.named_modules():
        #     if getattr(module, "qconfig", None) is not None:
        #         print(f"{name}: {module.qconfig}")

        # print(prepared_model)

        prepared_model.eval()
        #calibration - so that the observers can see the range of values
        with torch.no_grad():
            prepared_model(images)
        

        quantized_model = convert_fx(prepared_model, qconfig_mapping=custom_qconfig_mapping)


        # for name, module in quantized_model.named_modules():
        #     print(name, "->", type(module))

        return  quantized_model






#this is a replacement for the GELU function in the Swin ViT
#This is much better to quantize than GELU
# class Swish(torch.nn.Module):
#     def forward(self,x):
#         return x * torch.sigmoid(x)
    

#The ViT contains a layer norm that is not quantizable - need to find a way to replace it with something that is similar in function
# class FusedLinear(nn.Module.LayerNorm):
#     def forward(self, x, dtype = torch.quint8):
#         return super().forward(x.float()).to(x.dtype)




#-------------------------------------------------------------------------YOLO methods-------------------------------------------------------------------------

    