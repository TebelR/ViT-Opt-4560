import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.quantization
import copy
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
        return quant_model
    
    def static_quant_class(self, model = None, q_level = torch.qint8):
        if model == None:
            print("No model passed into the static quant function, deep copies of the ViT are not supported")
            return
        model.to("cpu")
        model.eval()
        #This mimics the fbgemm config but applies the specified dtype for quantization (q_level)
        # config = torch.quantization.QConfig(
        #     activation=torch.quantization.MinMaxObserver.with_args(reduce_range = True, dtype=q_level),
        #     weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=q_level, qscheme=torch.per_channel_symmetric)
        # )
        config = torch.quantization.get_default_qconfig('x86')
        model.qconfig = config
        model.features[0][0].qconfig = None
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) or isinstance(module, nn.Linear):
                module.qconfig = None
        print("Ommited layers: " + str(model.features[0][0]))
        model = torch.quantization.prepare(model)

        #Now that a bunch of observes have been attached to the model, it needs to be calibrated by feeding it some pictures
        #pick 100 random images from the validation set
        images = []
        labels = []
        for i in range(100):#get 100 random images and their labels
            random_pick = random.randint(0, len(self.dls.dl_validate_classification) - 1)
            image, label = self.dls.dataset_classification[random_pick]
            images.append(image)
            labels.append(label)

        device = ("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels).to(device)
        model.to(device)
        with torch.no_grad(): 
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

        output = torch.quantization.convert(model)
        return output
    

    def convert_class_biases(self, model, dtype = torch.float32):
        if model == None:
            print("No model passed into the bias conversion function for the classifier, deep copies of the ViT are not supported")
            return
        model.to("cpu")
        model.eval()
        for name, param in model.named_parameters():
            if "bias" in name:
                param.data = param.data.to(dtype)
        print("Biases converted to {}".format(dtype))
        return
        
        




    #This will make a deep copy of the classifier and will substitute some parts of it with quantized versions
    def prep_class_model(self):
        # subject = copy.deepcopy(self.orig_class_model)
        # subject.eval()
        # for module in subject.modules():
        pass
    



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

    