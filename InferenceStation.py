import random
import torch
import os
import json
import time
from torchvision import transforms as T
from ultralytics import YOLO
from DataLoadingStation import DataLoadingStation
from torchvision import io
import cv2
from sklearn.metrics import precision_score, recall_score
from timeit import default_timer as timer 
import torchvision.ops as ops

class InferenceStation():
    device = "cpu"
    record_mem_use = False
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

        with torch.no_grad():
            imgTemp = transform(image.to(torch.float32)/255.0).unsqueeze(0)
            imgTemp = imgTemp.to(self.device)
            if self.record_mem_use: 
                print(f"Before inference on ViT on a seed: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            output = model(imgTemp)
            if self.record_mem_use: 
                print(f"After inference on ViT on a seed: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                self.record_mem_use = False
            #map the existing classes to the output to see which class names are predicted
            classNameIndices = dls.dataset_classification.indexed_classes
            classNamePredictions = []
            output = torch.nn.functional.softmax(output, dim=1).cpu()
            top5 = (output.topk(5, dim=1).indices[0], output.topk(5, dim=1).values[0])

            for i, p in zip(top5[0], top5[1]):
                reverse_classNameIndices = {v: k for k, v in classNameIndices.items()}
                classNamePredictions.append((reverse_classNameIndices[i.item()], p.item()))
            #print("Results of classification: " + str(classNamePredictions))
            return classNamePredictions
        

    #this is used for the t-test and quantization
    #this will classify a batch of random labeled images from the same dataset this was trained on and will capture:
    #accuracy, precision, recall and f1 score
    # in a tuple (accuracy, precision, recall, f1)
    def inferOnClassificationAvg(self, model, dls:DataLoadingStation, device = "cpu", input_dtype = torch.float32):

        # images = []
        # labels = []
        precision = 0
        recall = 0
        correct = 0
        # num_tests = len(dls.dl_validate_classification)
        # for i in range(num_tests):
        #     random_pick = random.randint(0, len(dls.dl_validate_classification) - 1)
        #     image, label = dls.dataset_classification[random_pick]
        #     images.append(image)
        #     labels.append(label)

        # images = torch.stack(images).to(device)
        # labels = torch.tensor(labels).to(device)
        

        total = 0
        model.to(device)
        model.eval()
        for images, labels in dls.dl_validate_classification:
            #need to convert the images to the correct input dtype if the model was quantized
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():#, torch.autocast(device_type='cuda', dtype=torch.float16):  # Disable gradient computation for validation
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # Get predicted class indices
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                precision += precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                recall += recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)

            #print("Correct: " + str(correct))
        accuracy = correct / total
        precision = precision / len(dls.dl_validate_classification)
        recall = recall / len(dls.dl_validate_classification)
        f1 = 2 * (precision * recall) / (precision + recall)

        return (accuracy, precision, recall, f1)


            
        
    
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
        print("Image dimensions: " + str(imageWidth) + " " + str(imageHeight))
        start_time = timer()
        
        self.record_mem_use = True#prepare to record memory usage on the ViT for one seed - it's the same for all seeds and does not stack
        
        if imageWidth < 1000 or imageHeight < 1000:
            imageCV2 = cv2.resize(imageCV2, (1000, 1000))
            rescaleFactorX = imageWidth / 1000
            rescaleFactorY = imageHeight / 1000
            #print("Rescale factors: " + str(rescaleFactorX) + " " + str(rescaleFactorY))

        with torch.no_grad():
            if(self.device.type == "cuda"):
                torch.cuda.synchronize()
                print(f"Before inference on YOLO: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #first, create bounding boxes from the image
            output = modelDetect(image_path)
            if(self.device.type == "cuda"):
                torch.cuda.synchronize()
                print(f"After inference on YOLO: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
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
                    #if(xyxyn[i][3].item() - xyxyn[i][1].item() > 0.05 and xyxyn[i][2].item() - xyxyn[i][0].item() > 0.05): #this may occasionally be needed if the YOLO misbehaves
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

        self.record_mem_use = True
        end_time = timer()
        print("Total infrence time: " + str(end_time - start_time))



    #Infer on all the synthetic data that exists in the data path and evaluate the results
    def inferOnSynthetic(self, data_path, modelDetect:YOLO, modelClass, dls:DataLoadingStation, outputPath):
        modelDetect.to(self.device)
        modelClass.to(self.device)
        modelDetect.eval()
        modelClass.eval()
        modelDetect.val(data = data_path, save_json = True)#this does a lot of nice validation for the YOLO model only, but is not flexible enough for the ViT.

        #once the YOLO has been validated, manually infer on synthetic data to test the ViT model
        #with torch.no_grad():
        
    def inferQuantizedDetection(self, model_path, data_path="data/detection/data.yaml"):
        model = YOLO(model_path)

        # Measure latency and accuracy
        start_time = time.time()
        metrics = model.val(data=data_path)
        end_time = time.time()

        # Compute additional metrics
        latency = metrics.speed['inference']  # ms per image
        fps = round(1000 / latency, 2)  # Frames per second
        model_size = round(os.path.getsize(model_path) / (1024 * 1024), 2)  # MB

        return {
            "F1 Score": metrics.box.f1,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "Recall": metrics.box.mr,
            "Latency (ms)": latency,
            "FPS": fps,
            "Model Size (MB)": model_size,
            "Total Time (s)": round(end_time - start_time, 2),
        }
    


#This will use a yaml formatted dataset to evaluate how well the YOLO and the ViT work together on synthetic data
#For this to work correctly, the yaml labels need to contain true classes of seeds - not just 0 for "seed"
    def evaluateCombined(self, data_path, modelDetect:YOLO, modelClass, dls:DataLoadingStation):
        total = 0
        precision_class = 0
        recall_class = 0
        correct_class = 0

        modelDetect.to('cpu')
        modelClass.to('cpu')
        modelDetect.eval()
        modelClass.eval()
        
        all_images = os.listdir(os.path.join(data_path, "images", "test"))
        all_labels = os.listdir(os.path.join(data_path, "labels_class", "test"))

        transform = T.Compose([
            T.Resize((192, 272)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for i in range(len(all_images)):
            image_path = os.path.join(data_path, "images", "test", all_images[i])
            label_path = os.path.join(data_path, "labels_class", "test", all_labels[i])
            #image = cv2.imread(image_path)
            
            #each element in the list contains [class, x, y, width, height] - x y width height are normalized w respect to totlal image size
            true_box_classes = torch.tensor([list(map(float, line.split())) for line in open(label_path)], dtype=torch.float32)
            true_boxes = true_box_classes[:, 1:]
            true_box_xyxy = box_cxcywh_to_xyxy(true_boxes)
            #no need to unravel the normalized coordinates - the predictions give normalized x y w h
            
            modelDetect.to(self.device)
            modelClass.to(self.device)
            with torch.no_grad():
                output = modelDetect(image_path)

                for result in output:#for every detected object - get iou and then work with the ViT
                    class_labels_stacked = []
                    class_images_stacked = []
                    predictions = result.boxes.xyxyn
                    predictions = predictions.to('cpu')
                    true_box_xyxy = true_box_xyxy.to('cpu')
                    iou_list = ops.box_iou(predictions, true_box_xyxy)
                    best_match_indices = torch.argmax(iou_list, dim=1)
                    best_true_match = true_box_classes[best_match_indices, 0]

                    class_image = io.decode_image(image_path,  mode="RGB")
                    for i in range(len(predictions)):
                        x = predictions[i][0].item() * class_image.shape[2]
                        y = predictions[i][1].item() * class_image.shape[1]
                        w = predictions[i][2].item() * class_image.shape[2] - x
                        h = predictions[i][3].item() * class_image.shape[1] - y
                        seed = class_image[:, int(y):int(y+h), int(x):int(x+w)]
                        
                        imgTemp = transform(seed.to(torch.float32)/255.0).unsqueeze(0)
                        class_images_stacked.append(imgTemp)
                        class_labels_stacked.append(best_true_match[i])


                    class_images_stacked = torch.cat(class_images_stacked, dim=0)
                    class_labels_stacked = torch.stack(class_labels_stacked)

                    class_images_stacked = class_images_stacked.to(self.device)
                    class_labels_stacked = class_labels_stacked.to(self.device)

                    outputs = modelClass(class_images_stacked)#need to unsqueeze because normally a data loader would do this
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    correct_class += (preds == class_labels_stacked).sum().item()
                    total += class_labels_stacked.size(0)
                    precision_class += precision_score(class_labels_stacked.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                    recall_class += recall_score(class_labels_stacked.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                
        #final metrics for the ViT
        accuracy = correct_class / total
        precision_class = precision_class / len(dls.dl_validate_classification)
        recall_class = recall_class / len(dls.dl_validate_classification)
        f1 = 2 * (precision_class * recall_class) / (precision_class + recall_class)

        metrics_yolo = modelDetect.val(data = os.path.join(data_path, "data.yaml"), save_json = False)
        print("Evaluation complete...")
        print("Ignore YOLO classification stats.")
        print(f"YOLO metrics:\n{metrics_yolo.results_dict}\n{metrics_yolo.speed}")
        print(f"\nViT metrics:\nAccuracy: {accuracy}\nPrecision: {precision_class}\nRecall: {recall_class}\nF1: {f1}")




def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)       
        