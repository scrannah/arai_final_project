import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18


class RubbishClassifier:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet18 = resnet18(weights=None)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, 3)  # change the last layer (fc) into a three classifier

        # Instantiate the model and move it to the device
        self.resnet18 = self.resnet18.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def run_model(self, frame):

        transformed_frame = self.transform(frame)
        transformed_frame = transformed_frame.unsqueeze(0)
        transformed_frame = transformed_frame.to(self.device)
        self.resnet18.eval()
        with torch.no_grad():
            prediction = self.resnet18(transformed_frame)

        return prediction


