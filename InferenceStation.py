import torch
import os
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


    

    #Performs inference on a single seed will output top 3 classes
    def inferOnClassification(self, image, model, dls:DataLoadingStation):
        model.to(self.device)
        model.eval()
        transform = T.Compose([
            T.Resize((80,80)),
            T.Normalize(mean=[0.634, 0.562, 0.498], std=[0.204, 0.241, 0.244])
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
            top3 = (output.topk(3, dim=1).indices[0], output.topk(3, dim=1).values[0])

            # for i, p in zip(top3[0], top3[1]):
            #     reverse_classNameIndices = {v: k for k, v in classNameIndices.items()}
            return classNamePredictions
        

    #Classifies a batch of random labeled images from the validation set of the classification dataset and will capture:
    #accuracy, precision, recall and f1 score
    # in a tuple (accuracy, precision, recall, f1)
    def inferOnClassificationAvg(self, model, dls:DataLoadingStation, device = "cpu", input_dtype = torch.float32):
        precision = 0
        recall = 0
        correct = 0        

        total = 0
        model.to(device)
        model.eval()
        for images, labels in dls.dl_validate_classification:
            images, labels = images.to(device), labels.to(device)
           
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # Get predicted class indices
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                precision += precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                recall += recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)

        accuracy = correct / total
        precision = precision / len(dls.dl_validate_classification)
        recall = recall / len(dls.dl_validate_classification)
        if(precision != 0 and recall != 0):
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            print("avoiding F1 division by 0")

        return (accuracy, precision, recall, f1)


            
        
    #Uses the YOLO to test coodinates of detected seeds. This does not draw anything on the image and should be used for debugging only
    def inferOnDetection(self, image_path, model):
        model.to(self.device)
        model.eval()
        print("Infering on detection")
        with torch.no_grad():
            output = model(image_path, show=True)
            for result in output:
                xywh = result.boxes.xywh  # center-x, center-y, width, height
                xywhn = result.boxes.xywhn  # normalized
                xyxyn = result.boxes.xyxyn  # normalized
                # confs = result.boxes.conf  # confidence score of each box
                print("\n Normalized: \n" + str(xyxyn))
                print("\n Centers: \n" + str(xywh))
                print("\n Normalized Centers: \n" + str(xywhn))
        


    #This will classify all seeds in an image located at image_path. Boxes and top 3 classes will be drawn for every seed in the image. The new image will be saved in outputPath
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
        
        if imageWidth < 1000 or imageHeight < 1000:#This needs reconfiguration to work with small and very large images
            imageCV2 = cv2.resize(imageCV2, (1000, 1000), interpolation =  cv2.INTER_AREA)
            rescaleFactorX = imageWidth / 1000
            rescaleFactorY = imageHeight / 1000

        with torch.no_grad():
            if(self.device.type == "cuda"):
                torch.cuda.synchronize()
                print(f"Before inference on YOLO: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #first, create bounding boxes from the image
            output = modelDetect(imageCV2) #img_path
            if(self.device.type == "cuda"):
                torch.cuda.synchronize()
                print(f"After inference on YOLO: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #since the input image resolution could've been resized to fit YOLO, normalized coordinates of boxes are used
            detectedSeeds = []
            for result in output:
                xyxyn = result.boxes.xyxyn  # normalized top-left-x, top-left-y, bottom-right-x, bottom-right-y
                confs = result.boxes.conf  # confidence score of each box
                for i in range(len(xyxyn)):
                    seedCrop = {
                        "top" : xyxyn[i][1].item(),
                        "left" : xyxyn[i][0].item(),
                        "bottom" : xyxyn[i][3].item(),
                        "right" : xyxyn[i][2].item(),
                        "confidence": confs[i]
                    }
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




    #Looks at the stats of the quantized YOLO model
    def inferQuantizedDetection(self, model_path, data_path="data/syntheticData/data.yaml"):
        model = YOLO(model_path)

        # measure latency and accuracy
        start_time = time.time()
        metrics = model.val(data=data_path)
        end_time = time.time()

        # compute additional metrics
        latency = metrics.speed['inference']  # ms per image
        fps = round(1000 / latency, 2)  # frames per second - if interested
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
            T.Resize((80,80)),
            T.Normalize(mean=[0.634, 0.562, 0.498], std=[0.204, 0.241, 0.244])
        ])
        
        for i in range(len(all_images)):
            image_path = os.path.join(data_path, "images", "test", all_images[i])
            label_path = os.path.join(data_path, "labels_class", "test", all_labels[i])
            
            #each element in the list contains [class, x, y, width, height] - x y width height are normalized with respect to totlal image size
            true_box_classes = torch.tensor([list(map(float, line.split())) for line in open(label_path)], dtype=torch.float32)
            true_boxes = true_box_classes[:, 1:]
            true_box_xyxy = box_cxcywh_to_xyxy(true_boxes)
            #no need to unravel the normalized coordinates - the predictions give normalized x y w h
            
            modelDetect.to(self.device)
            modelClass.to(self.device)
            with torch.no_grad():
                output = modelDetect(image_path)
                
                batches = 0
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
                        imgTemp = transform(seed.to(torch.float32)/255.0).unsqueeze(0)#normalize and convert to tensor
                        
                        class_images_stacked.append(imgTemp)
                        class_labels_stacked.append(best_true_match[i])


                    class_images_stacked = torch.cat(class_images_stacked, dim=0)
                    class_labels_stacked = torch.stack(class_labels_stacked)

                    class_images_stacked = class_images_stacked.to(self.device)
                    class_labels_stacked = class_labels_stacked.to(self.device)

                    outputs = modelClass(class_images_stacked)
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    correct_class += (preds == class_labels_stacked).sum().item()
                    total += class_labels_stacked.size(0)
                    precision_class += precision_score(class_labels_stacked.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                    recall_class += recall_score(class_labels_stacked.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
                    batches+=1
                
        #final metrics for the ViT
        accuracy = correct_class / total
        precision_class = precision_class / (batches * len(all_images))
        recall_class = recall_class / (batches * len(all_images))
        f1 = 2 * (precision_class * recall_class) / (precision_class + recall_class)

        metrics_yolo = modelDetect.val(data = os.path.join(data_path, "data.yaml"), save_json = False)
        print("Evaluation complete...")
        print("Ignore YOLO classification stats.")
        print(f"YOLO metrics:\n{metrics_yolo.results_dict}\n{metrics_yolo.speed}")
        print(f"\nViT metrics:\nAccuracy: {accuracy}\nPrecision: {precision_class}\nRecall: {recall_class}\nF1: {f1}")




#Helper method to convert center x, center y, width, height to x1, y1, x2, y2 of the bounding box - useful for cropping out the plant seeds for ViT
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)       
        