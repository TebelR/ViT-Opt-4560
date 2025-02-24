import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
from torchinfo import summary
from torchvision.models import swin_v2_t
from timeit import default_timer as timer 
import os
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
    criterion = nn.CrossEntropyLoss()
    optimizer = None #optim.Adam(model.parameters(), lr=learning_rate)

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

    



    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()         # Clear previous gradients
                outputs = self.model(images)       # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                loss.backward()               # Backpropagation: compute gradients
                self.optimizer.step()              # Update model parameters
                
                running_loss += loss.item() * images.size(0)  # Accumulate batch loss

            epoch_loss = running_loss / len(self.train_dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            # Validation phase
            self.model.eval()  # Switch to evaluation mode (disables dropout, etc.)
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient computation for validation
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, preds = torch.max(outputs, 1)  # Get predicted class indices
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")

        end_time = timer()

        print(f"[INFO] Total training time: {end_time-self.start_time:.3f} seconds")



        
