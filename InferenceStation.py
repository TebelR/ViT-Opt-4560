import torch
import os
import json
from torchvision import transforms as T




class InferenceStation():
    device = "cpu"
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    

    #Classification will output top 5 classes
    def inferOnClassification(self, image, model):
        model.to(self.device)
        model.eval()
        transform = T.Compose([
            T.Resize((192, 272)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Infering on classification")
        #print("Image shape: " + str(image.shape))
        with torch.no_grad():
            imgTemp = transform(image.to(torch.float32)/255).unsqueeze(0)
            imgTemp = imgTemp.to(self.device)
            output = model(imgTemp)
            torch.sort(output, descending=True)
            return output[0:4]
        
    
    def inferOnDetection(self, image_path, model):
        model.to(self.device)
        model.eval()
        print("Infering on detection")
        #print("Image path: " + str(image_path))
        # transform = T.Compose([
        #     T.Resize((640,640))
        # ])
        with torch.no_grad():
            #image = transform(image).unsqueeze(0).to(self.device)
            output = model(image_path)
            return output
        

    def inferOnCombined(self, image_path, modelDetect, modelClass):
        modelDetect.to(self.device)
        modelClass.to(self.device)
        modelDetect.eval()
        modelClass.eval()
        with torch.no_grad():
            #first, create bounding boxes from the image
            #image = image.to(self.device)
            output = modelDetect(image_path)
            #Then, crop out each seed based on its bounding box and feed that into the classification model
            print(output)
            