import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18


class SmartTransform:  # smart transform to only crop images that need it
    def __init__(self):
        self.center_crop = transforms.CenterCrop(224)
        self.resize = transforms.Resize((64, 64))

    def __call__(self, frame):
        w, h = frame.size
        # Only crop if image is big enough
        if min(w, h) >= 224:
            frame = self.center_crop(frame)
        frame = self.resize(frame)
        return frame


class RubbishClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet18 = resnet18(weights=None)
        self.resnet18.load_state_dict(torch.load("path here"))
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, 3)  # change the last layer (fc) into a three classifier

        # Instantiate the model and move it to the device
        self.resnet18 = self.resnet18.to(self.device)
        self.resnet18.eval()

        self.transform = transforms.Compose([
            SmartTransform(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def run_model(self, frame):
        transformed_frame = self.transform(frame)
        transformed_frame = transformed_frame.unsqueeze(0)  # add batch dim on even though im only passing 1 frame
        transformed_frame = transformed_frame.to(self.device)
        with torch.no_grad():
            classification = self.resnet18(transformed_frame)

        return classification
