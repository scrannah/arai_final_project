import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from collections import Counter

class SmartTransform: # smart transform to only crop images that need it
    def __init__(self):
        self.center_crop = transforms.CenterCrop(224)
        self.resize = transforms.Resize((64, 64))
    def __call__(self, img):
        w, h = img.size
        # Only crop if image is big enough
        if min(w, h) >= 224:
            img = self.center_crop(img)
        img = self.resize(img)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Class names for dataset
class_names = ['metal', 'wood', 'cardboard']

class RubbishClassifier
original_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(original_model)
in_features = original_model.fc.in_features
original_model.fc = nn.Linear(in_features, 3) #change the last layer (fc) into a three classifier

# Instantiate the model and move it to the device
model=original_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4)

def classifier(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy, all_labels, all_predictions