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


class SmartTransform:  # smart transform to only crop images that need it
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


class RubbishClassifier:

    def __init__(self):
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.in_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(in_features, 3)  # change the last layer (fc) into a three classifier

        # Instantiate the model and move it to the device
        self.model = resnet18.to(device)


    def run_model(self, resnet18, frame):
        resnet18.eval
        with torch.no_grad():
            prediction = self.resnet18(frame)

        return prediction