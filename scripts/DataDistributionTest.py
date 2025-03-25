#This script will calculate the mean and standard deviation of the classification dataset for normalization
import torch
from torchvision import datasets, transforms
from DataLoadingStation import DataLoadingStation

dls = DataLoadingStation()
dls.load_data_classification(0.99, 0.01)

mean = 0
std = 0
num_batches = 0

for images, _ in dls.dl_train_classification:
    batch_samples = images.size(0)  # Batch size
    images = images.view(batch_samples, 3, -1)  # Flatten HxW into one dimension
    mean += images.mean(dim=[0, 2])  # Mean over batch and spatial dims
    std += images.std(dim=[0, 2])  # Std over batch and spatial dims
    num_batches += 1

mean /= num_batches
std /= num_batches

print(f"Dataset Mean: {mean.tolist()}")
print(f"Dataset Std: {std.tolist()}")