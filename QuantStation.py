import random
import torch
import os
import torch.ao.quantization as quant
from torch.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from torch.ao.quantization import get_default_qconfig_mapping, prepare, convert, HistogramObserver, MinMaxObserver, QConfig, PerChannelMinMaxObserver, FakeQuantize
import torch.ao.quantization.qconfig_mapping as qc_mapping
from DataLoadingStation import DataLoadingStation
from InferenceStation import InferenceStation
import torch.ao.quantization.qconfig as qconfig
from DataLoadingStation import DataLoadingStation
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



#Dynamically quantize the classifier. This really only supports int8 quantization as the Swin ViT does not behave well when half() is called on it. 
#After our trials we learned that torch.autocast does not work with the ViT either - barely anything gets quantized this way as Pytorch itself detects that
#float16 conversion will blow things up in the ViT.
    def dynamic_quant_class(self, dtype_incoming = None, model = None):
        if model == None:
            print("No model passed into the dynamic quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")
        model.eval()
        
        
        custom_qconfig = qconfig.default_qconfig
        conv_qconfig = qconfig.default_per_channel_qconfig
        if dtype_incoming == torch.float16:#Using this here will throw an error
            conv_qconfig = qconfig.per_channel_dynamic_qconfig
            custom_qconfig = qconfig.float16_dynamic_qconfig
        
        #this is a dictionary that will hold the qconfig for each module
        #The modules of a model can be viewed by printing the model
        config_dict = {
            "features.0": None,
            "features.0.0": conv_qconfig,

            "features.1": None,#this is a swin transformer block
            #"features.1.0.attn.proj": qconfig.default_qconfig,#config for one of the Linear modules in the first Swin block
            "features.1.0.mlp.0": custom_qconfig,#linear
            "features.1.0.mlp.3": custom_qconfig,#linear

            #"features.1.1.atn.proj": qconfig.default_qconfig,#linear attention
            "features.1.1.mlp.0": custom_qconfig,#linear
            "features.1.1.mlp.3": custom_qconfig,#linear

            "features.2": None,# this is a PatchMergingV2 - no touch

            "features.3": None,#this is a swin transformer block
            #"features.3.0.attn.proj": qconfig.default_qconfig,#config for one of the Linear modules in the first Swin block
            "features.3.0.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.3.0.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.3.0.mlp.0": custom_qconfig,#linear
            "features.3.0.mlp.3": custom_qconfig,#linear

           #"features.3.1.atn.proj": qconfig.default_qconfig,#linear attention
            "features.3.1.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.3.1.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.3.1.mlp.0": custom_qconfig,#linear
            "features.3.1.mlp.3": custom_qconfig,#linear

            "features.4": None,#this is a PatchMergingV2 - no touch

            "features.5": None,#this one contains many transformer blocks
            #"features.5.0.attn.proj": qconfig.default_qconfig,#linear
            "features.5.0.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.5.0.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.5.0.mlp.0": custom_qconfig,#linear
            "features.5.0.mlp.3": custom_qconfig,#linear

            #"features.5.1.attn.proj": qconfig.default_qconfig,#linear
            "features.5.1.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.5.1.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.5.1.mlp.0": custom_qconfig,#linear
            "features.5.1.mlp.3": custom_qconfig,#linear

            #"features.5.2.attn.proj": qconfig.default_qconfig,#linear
            "features.5.2.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.5.2.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.5.2.mlp.0": custom_qconfig,#linear
            "features.5.2.mlp.3": custom_qconfig,#linear

            #"features.5.3.attn.proj": qconfig.default_qconfig,#linear
            "features.5.3.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.5.3.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.5.3.mlp.0": custom_qconfig,#linear
            "features.5.3.mlp.3": custom_qconfig,#linear

            #"features.5.4.attn.proj": qconfig.default_qconfig,#linear
            "features.5.4.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.5.4.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.5.4.mlp.0": custom_qconfig,#linear
            "features.5.4.mlp.3": custom_qconfig,#linear

            #"features.5.5.attn.proj": qconfig.default_qconfig,#linear
            "features.5.5.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU            
            "features.5.5.attn.cpb_mlp.2": custom_qconfig,#linear            
            "features.5.5.mlp.0": custom_qconfig,#linear
            "features.5.5.mlp.3": custom_qconfig,#linear

            "features.6": None,#this is a PatchMergingV2

            "features.7": None,#this has two transformer blocks
            #"features.7.0.attn.proj": qconfig.default_qconfig,#linear
            "features.7.0.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.7.0.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.7.0.mlp.0": custom_qconfig,#linear
            "features.7.0.mlp.3": custom_qconfig,#linear

            #"features.7.1.attn.proj": qconfig.default_qconfig,#linear
            "features.7.1.attn.cpb_mlp.0": custom_qconfig,#linear - index 1 is a ReLU
            "features.7.1.attn.cpb_mlp.2": custom_qconfig,#linear
            "features.7.1.mlp.0": custom_qconfig,#linear
            "features.7.1.mlp.3": custom_qconfig,#linear

            "norm": None,
            "avgpool": custom_qconfig, #avg pool can be quantized
            "flatten": None,
            "head": custom_qconfig,# this is a linear too
        }

        #this propagates all the qconfigs and places the observers in the right places - scaling factors will be calculated at runtime and the model will keep track of them
        quant_model = torch.quantization.quantize_dynamic(model, qconfig_spec=config_dict, inplace = False)

        return quant_model
    
            
        

    
    def static_quant_class(self, model = None, dls : DataLoadingStation = None, dtype_incoming = torch.quint8):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")#move the model to the cpu as it may misbehave on the gpu
        model.eval()

        num_calibration_imgs = 100
        images = []
        for i in range(num_calibration_imgs):
            random_choice = random.randint(0, len(dls.dataset_classification) - 1)
            image, _ = dls.dataset_classification[random_choice]
            images.append(image)

        
        images = torch.stack(images).to("cpu")

        #define a custom qconfig for linear modules
        linear_custom_qconfig = QConfig(
            activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min = 0, quant_max = 255),
            weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
        )
        
        #also a custom qconfig for Conv2D modules
        conv2d_custom_qconfig = QConfig(
            activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine, bins=64, reduce_range = True),
            weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
        
        custom_global_qconfig = QConfig(
            activation = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
            weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )

        custom_qconfig_mapping = get_default_qconfig_mapping("fbgemm")#qc_mapping.QConfigMapping()
        #custom_qconfig_mapping.set_global(custom_global_qconfig)#ignore everything except for certain modules defined below

        for name, module in model.features.named_modules():#fuse the linear and ReLU for better performance
            if isinstance(module, SwinTransformerBlockV2):
                quant.fuse_modules(module, ["attn.cpb_mlp.0", "attn.cpb_mlp.1"], inplace=True)

        for name, module in model.named_modules():#then apply custom qconfigs to selected modules - this is in a separate loop because otherwise there's a recursion overflow
            if isinstance(module, torch.nn.Conv2d):
                custom_qconfig_mapping.set_module_name(name, conv2d_custom_qconfig)
                
            elif isinstance(module, torch.nn.Linear):#do not touch qkv and head linear layers
                custom_qconfig_mapping.set_module_name(name, linear_custom_qconfig)
            else:
                custom_qconfig_mapping.set_module_name(name, None)#make sure that nothing else gets quantized

        noise_input = torch.randn(1, 3, 80, 80).to("cpu")
        #converts the model to a graph by symbolic tracing (from noise input)
        prepared_model = prepare_fx(model=model, qconfig_mapping=custom_qconfig_mapping, example_inputs=noise_input)


        prepared_model.eval()
        #calibration - so that the observers can see the range of values
        with torch.no_grad():
            prepared_model(images)
        
        #quantizes the graph
        quantized_model = convert_fx(prepared_model, qconfig_mapping=custom_qconfig_mapping)

        return  quantized_model
    




    #Simple naive approach with default values
    def static_graph_quant_class(self, model = None, dls : DataLoadingStation = None):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")#move the model to the cpu as it may misbehave on the gpu
        

        num_calibration_imgs = 100
        images = []
        for i in range(num_calibration_imgs):
            random_choice = random.randint(0, len(dls.dataset_classification) - 1)
            image, _ = dls.dataset_classification[random_choice]
            images.append(image)

        
        images = torch.stack(images).to("cpu")
        qconfig_mapping = get_default_qconfig_mapping("fbgemm")
        prepared_model = prepare_fx(model, qconfig_mapping= qconfig_mapping, example_inputs=images)
        
        #calibration
        model.eval()
        with torch.no_grad():
            prepared_model(images)

        quantized_model = convert_fx(prepared_model, qconfig_mapping= qconfig_mapping)

        return  quantized_model

#There is no YOLO quantization here as ultralytics allows to export and quantize the YOLO with one line of code.
    