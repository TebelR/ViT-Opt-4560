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
            model = self.orig_class_model
        
        subject = copy.deepcopy(model)
        subject.to("cpu")
        subject.eval()
        
        quant_model = torch.quantization.quantize_dynamic(subject, {nn.Linear}, dtype=dtype)
        return quant_model
    
    def static_quant_class(self, model = None, q_level = torch.qint8):
        if model == None:
            model = self.orig_class_model
        subject = copy.deepcopy(model)
        subject.to("cpu")
        subject.eval()
        #This mimics the fbgemm config but applies the specified dtype for quantization (q_level)
        config = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(reduce_range = True, dtype=q_level),
            weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=q_level, qscheme=torch.per_channel_symmetric)
        )
        subject.qconfig = config
        subject = torch.quantization.prepare(subject)

        #Now that a bunch of observes have been attached to the model, it needs to be calibrated by feeding it some pictures
        #pick 10 random images from the validation set
        images = []
        labels = []
        for i in range(100):#get 100 random images and their labels
            random_pick = random.randint(0, len(self.dls.dl_validate_classification) - 1)
            image, label = self.dls.dataset_classification[random_pick]
            images.append(image)
            labels.append(label)

        images = torch.stack(images).to(self.dev)
        labels = torch.tensor(labels).to(self.dev)
        with torch.no_grad(): 
            outputs = subject(images)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices

        subject = torch.quantization.convert(subject)
        return subject
        
        




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

    