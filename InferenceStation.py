import torch
import os
import json
from torchvision import transforms as T
from ultralytics import YOLO
from DataLoadingStation import DataLoadingStation
from torchvision import io
import cv2

class InferenceStation():
    device = "cpu"
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    

    #Classification will output top 5 classes
    def inferOnClassification(self, image, model, dls:DataLoadingStation):
        model.to(self.device)
        model.eval()
        transform = T.Compose([
            T.Resize((192, 272)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #print("Infering on classification")
        #print("Image shape: " + str(image.shape))
        with torch.no_grad():
            #print("Image shape prior to crop: " + str(image.shape))
            imgTemp = transform(image.to(torch.float32)/255.0).unsqueeze(0)
            imgTemp = imgTemp.to(self.device)
            output = model(imgTemp)
            #map the existing classes to the output to see which class names are predicted
            classNameIndices = dls.dataset_classification.indexed_classes
            classNamePredictions = []
            output = torch.nn.functional.softmax(output, dim=1).cpu()
            top5 = (output.topk(5, dim=1).indices[0], output.topk(5, dim=1).values[0])
            #print("Top 5 classes: " + str(top5))

            for i, p in zip(top5[0], top5[1]):
                reverse_classNameIndices = {v: k for k, v in classNameIndices.items()}
                classNamePredictions.append((reverse_classNameIndices[i.item()], p.item()))
            #print("Results of classification: " + str(classNamePredictions))
            return classNamePredictions
            
        
    
    def inferOnDetection(self, image_path, model):
        model.to(self.device)
        model.eval()
        print("Infering on detection")
        with torch.no_grad():
            #image = transform(image).unsqueeze(0).to(self.device)
            output = model(image_path, show=True)
            for result in output:
                xywh = result.boxes.xywh  # center-x, center-y, width, height
                xywhn = result.boxes.xywhn  # normalized
                #xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
                xyxyn = result.boxes.xyxyn  # normalized
                #names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                confs = result.boxes.conf  # confidence score of each box
                #print("Results of detection: \n" + str(xyxy) + "\n" + str(names) + "\n" + str(confs))
                print("\n Normalized: \n" + str(xyxyn))
                print("\n Centers: \n" + str(xywh))
                print("\n Normalized Centers: \n" + str(xywhn))
        

    def inferOnCombined(self, image_path, modelDetect, modelClass, dls:DataLoadingStation, outputPath):
        modelDetect.to(self.device)
        modelClass.to(self.device)
        modelDetect.eval()
        modelClass.eval()
        rescaleFactorX = 1
        rescaleFactorY = 1
        imageCV2 = cv2.imread(image_path)
        imageWidth, imageHeight, _ = imageCV2.shape

        if imageWidth < 1000 or imageHeight < 1000:
            imageCV2 = cv2.resize(imageCV2, (1000, 1000))
            rescaleFactorX = imageWidth / 1000
            rescaleFactorY = imageHeight / 1000
            #print("Rescale factors: " + str(rescaleFactorX) + " " + str(rescaleFactorY))

        with torch.no_grad():
            #first, create bounding boxes from the image
            output = modelDetect(image_path)
            #since the input image resolution could've been resized to fit YOLO, normalized coordinates of boxes are used
            detectedSeeds = []
            for result in output:
                #xywhn = result.boxes.xywhn  # normalized center-x, center-y, width, height
                xyxyn = result.boxes.xyxyn  # normalized top-left-x, top-left-y, bottom-right-x, bottom-right-y
                #names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                confs = result.boxes.conf  # confidence score of each box
                for i in range(len(xyxyn)):
                    seedCrop = {
                        "top" : xyxyn[i][1].item(),
                        "left" : xyxyn[i][0].item(),
                        "bottom" : xyxyn[i][3].item(),
                        "right" : xyxyn[i][2].item(),
                        "confidence": confs[i]
                    }
                    if(xyxyn[i][3].item() - xyxyn[i][1].item() > 0.05 and xyxyn[i][2].item() - xyxyn[i][0].item() > 0.05):
                        print("appended a seed crop: " + str(seedCrop))
                        detectedSeeds.append(seedCrop)
                
                image = io.decode_image(image_path,  mode="RGB")
                #now, crop out another image from the original image and classify it - this is done for every detected seed
                for i in range(len(detectedSeeds)):
                    seedCrop = detectedSeeds[i]
                    #load the original image
                    #crop the image to the seed
                    x = float(seedCrop["left"]) * image.shape[2]
                    y = float(seedCrop["top"]) * image.shape[1]
                    w = float(seedCrop["right"]) * image.shape[2] - x
                    h = float(seedCrop["bottom"]) * image.shape[1] - y
                    seed = image[:, int(y):int(y+h), int(x):int(x+w)]
                    #classify the seed
                    preds = self.inferOnClassification(seed, modelClass, dls)
                    #draw the box, confidence and labels for the seed
                    cv2.rectangle(imageCV2, (round(x/rescaleFactorX), round(y/rescaleFactorY)), (round((x+w)/rescaleFactorX), round((y+h)/rescaleFactorY)), (0, 255, 0), 2)
                    cv2.putText(imageCV2, f"Conf: {seedCrop['confidence']:.02f}", (round(x/rescaleFactorX), round((y)/rescaleFactorY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for j in range(len(preds)):
                        cv2.putText(imageCV2, f"{preds[j][0]}: {preds[j][1]:.02f}", (round(x/rescaleFactorX), round((y+h)/rescaleFactorY) + (j+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(outputPath, imageCV2)



    #Infer on all the synthetic data that exists in the data path and evaluate the results
    def inferOnSynthetic(self, data_path, modelDetect:YOLO, modelClass, dls:DataLoadingStation, outputPath):
        modelDetect.to(self.device)
        modelClass.to(self.device)
        modelDetect.eval()
        modelClass.eval()
        modelDetect.val(data = data_path, save_json = True)#this does a lot of nice validation for the YOLO model only, but is not flexible enough for the ViT.

        #once the YOLO has been validated, manually infer on synthetic data to test the ViT model
        #with torch.no_grad():
        
                

                    
        