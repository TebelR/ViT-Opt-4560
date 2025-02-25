import json
import random
# import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms, models
from torchvision import transforms as T
# from torchinfo import summary
from torchvision.models import swin_v2_t
from timeit import default_timer as timer 
import os
from torch.optim.lr_scheduler import StepLR
import tqdm
from sklearn.metrics import precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
# device agnostic code
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained weights
#weights = Swin_V2_T_Weights.IMAGENET1K_V1

# Create the model with pretrained weights
#model = swin_v2_t(weights=weights).to(device)

# # Print a summary using torchinfo (uncomment for actual output)
# summary(model=model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

# Training transformations with augmentation
# train_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),  # Data augmentation to improve generalization
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225])
# ])

# # Validation/test transformations without augmentation
# test_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                          std=[0.229, 0.224, 0.225])
# ])

# train_dataset = datasets.ImageFolder(root='data/visionDataTemp/train', transform=train_transforms)
# test_dataset  = datasets.ImageFolder(root='data/visionDataTemp/test',  transform=test_transforms)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# num_classes = len(train_dataset.classes)

# for param in model.parameters():
#     param.requires_grad = False

# in_features = model.head.in_features
# model.head = nn.Linear(in_features, num_classes)

# Set parameters
# learning_rate = 0.0001
# num_epochs = 1
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# summary(model=model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )

# start_time = timer()

# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()         # Clear previous gradients
#         outputs = model(images)       # Forward pass
#         loss = criterion(outputs, labels)  # Compute loss
#         loss.backward()               # Backpropagation: compute gradients
#         optimizer.step()              # Update model parameters
        
#         running_loss += loss.item() * images.size(0)  # Accumulate batch loss

#     epoch_loss = running_loss / len(train_dataset)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

#     # Validation phase
#     model.eval()  # Switch to evaluation mode (disables dropout, etc.)
#     correct = 0
#     total = 0
#     with torch.no_grad():  # Disable gradient computation for validation
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)  # Get predicted class indices
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# end_time = timer()

# print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

class ClassifierTrainer:

    num_classes = 0
    test_set = None
    train_set = None
    valid_set = None

    train_dl = None
    valid_dl = None
    test_dl = None
    
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.dirname(__file__)
    w_save_dir = os.path.join(project_root, "runs/classify")
    results_save_dir = os.path.join(project_root, "results/classification")

    start_time = timer()

    learning_rate = 0.0001
    num_epochs = 1
    num_instances = 5#run through each batch 5 tiems
    criterion = nn.CrossEntropyLoss()
    optimizer = None #optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None

    #this variable is used to store runtime training data
    training_data = []# for each epoch, store loss, accuracy, learning rate, precision, recall, time_to_train
    #this stores the lowest loss weights and the last epoch weights that were trained - after training this should have 2 elements
    weights = {
        "best_weights" : None,
        "last_weights" : None
    }

    #This assumes that the model has the weights that it wants to use prior to instantiating this class
    def __init__(self, train_dl : DataLoader, valid_dl : DataLoader, test_dl : DataLoader, model : swin_v2_t, num_classes):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.model = model

        self.num_classes = num_classes

        self.train_set = train_dl.dataset
        self.valid_set = valid_dl.dataset
        self.test_set = test_dl.dataset

        for param in model.parameters():
            param.requires_grad = False

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
    



    def train(self):
        self.model.train()  # Set model to training mode
        total_start_time = timer()
        best_f1 = 0# to keep track of the best model weights - F1 is a good metric since some plant species have very few images in the dataset - this prevents dominance
        writer = SummaryWriter(os.path.join(self.results_save_dir, "tsBoard"))
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            start_time = timer()
            avg_loss = 0
            for i in tqdm(range(len(self.train_dl))):
                losses = []

                transform = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(random(1, 359)),
                    T.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            

                images, labels = next(iter(self.train_dl))

                for i in range(self.num_instances):#run through multiple instances with different augmentations applied to the batch

                    for image in images:
                        image = transform(image)# apply random transformations
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)#THIS MAY NOT WORK
                    losses.append(loss.item())
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices

                avg_loss = sum(losses) / len(losses)
                (nn.CrossEntropyLoss)(avg_loss).backward()#This may not work either
                self.optimizer.step()

            self.scheduler.step()#move the learning rate
            # Validation phase
            self.model.eval()  # Switch to evaluation mode (disables dropout, etc.)
            correct = 0
            total = 0
            epoch_accuracy = 0
            precision = 0
            recall = 0

            with torch.no_grad():  # Disable gradient computation for validation
                for images, labels in self.test_dl:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    precision += precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                    recall += recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            epoch_accuracy = 100 * correct / total
            learning_rate = self.scheduler.get_last_lr()
            precision = precision / len(self.test_dl)
            recall = recall / len(self.test_dl)
            f1 = 2 * (precision * recall) / (precision + recall)

            writer.add_scalar("Loss", avg_loss, epoch)
            writer.add_scalar("Accuracy", epoch_accuracy, epoch)
            writer.add_scalar("Precision", precision, epoch)
            writer.add_scalar("Recall", recall, epoch)
            writer.add_scalar("F1", f1, epoch)
            writer.add_scalar("Learning Rate", learning_rate[0], epoch)

            #track the best model weigths
            if(f1 > self.best_f1):
                self.best_f1 = f1
                self.weights["best"] = self.model.state_dict()
            end_time = timer()

            #group epoch data
            epoch_data = {
                "epoch" : epoch,
                "loss" : avg_loss,
                "accuracy" : epoch_accuracy,
                "precision" : precision,
                "recall" : recall,
                "learning_rate" : learning_rate,
                "f1" : f1,
                "time" : end_time-start_time
            }
            
            self.training_data.append(epoch_data)
            self.weights["last"] = self.model.state_dict()
            print(f"Loss: {sum(losses)/len(losses):.4f}, Accuracy: {epoch_accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, Learning Rate: {learning_rate[0]:.4f}, Time: {end_time-start_time:.2f} seconds")

        total_end_time = timer()
        print("Training complete. Saving weigths...")
        print(f"[INFO] Total training time: {total_end_time-total_start_time:.3f} seconds")
        
        self.save_training_data()
        writer.close()




    def save_training_data(self):
        data = None
        with open(self.project_root + "variables.json") as f:
            data = json.load(f)
        train_index = data["best_classification_index"]
        data["best_classification_index"] += 1

        run_w_path = os.path.join(self.w_save_dir, "training_data_" + str(train_index))
        os.makedirs(run_w_path, exist_ok=True)
        torch.save(self.weights["best"], os.path.join(run_w_path, "best.pt"))
        torch.save(self.weights["last"], os.path.join(run_w_path, "last.pt"))


        run_results_path = os.path.join(self.results_save_dir, "training_data_" + str(train_index))
        os.makedirs(self.results_save_dir, exist_ok=True)

        with open(run_results_path + "results.json", "w") as f:
            json.dump(self.training_data, f)
        
        with open(self.project_root + "variables.json", "w") as f:
            json.dump(data, f)
        
    



        
