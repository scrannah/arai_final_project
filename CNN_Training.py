import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class names for dataset
class_names = ['metal', 'wood', 'cardboard']

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

        return img

class DatasetWrapper(Dataset):
    def __init__(self, base_dataset, indices, transform):
        # Pass args for each dataset instead of making wrapper for each
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        img = self.transform(img)
        return img, label

def get_transforms(stage):
    if stage == "stage1":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            SmartTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif stage == "stage2":
        train_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05),
            transforms.RandomRotation(20),
            SmartTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif stage == "stage3":
        train_transform = transforms.Compose([
            SmartTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    eval_transform = transforms.Compose([
            SmartTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    return train_transform, eval_transform


def split_dataset(base_dataset):
    labels = torch.tensor(base_dataset.targets)
    indices = torch.arange(len(base_dataset))

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=42)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_y,
        random_state=42)

    return train_idx, val_idx, test_idx

def count_classes(base_dataset, indices):
    labels = [base_dataset[i][1] for i in indices]
    return Counter(labels)

def build_dataloaders(dataset_path, stage, batch_size=16):
    base_dataset = ImageFolder(dataset_path)

    train_idx, val_idx, test_idx = split_dataset(base_dataset)
    print("TRAIN:", count_classes(base_dataset, train_idx))
    print("VAL:  ", count_classes(base_dataset, val_idx))
    print("TEST: ", count_classes(base_dataset, test_idx))
    train_transform, eval_transform = get_transforms(stage)

    train_set = DatasetWrapper(base_dataset, train_idx, train_transform)
    val_set   = DatasetWrapper(base_dataset, val_idx, eval_transform)
    test_set  = DatasetWrapper(base_dataset, test_idx, eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)



    return train_loader, val_loader, test_loader

original_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(original_model)
in_features = original_model.fc.in_features
original_model.fc = nn.Linear(in_features, 3) #change th last layer (fc) into a three classifier

# alternate version if wanted
class ResNet18Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

# model = ResNet18Small() if i wanted a modified resnet

# Instantiate the model and move it to the device
model=original_model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.1e-4)

train_loader, val_loader, test_loader = build_dataloaders(
    "C:\\Users\\hanna\\Downloads\\dataset_stage1_real", # edit for stage dataset
    stage="stage1",
    batch_size=4)

# Training loop

def train_epoch(model, dataloader, optimiser, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy


def validate(model, dataloader, criterion, device):
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


num_epochs = 15

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimiser, criterion, device)

    # Unpack all four return values, even if all_labels and all_predictions are not used within the loop
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} Val   Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "firstmodel.pth")
print("Model saved")


val_loss, val_acc, all_labels, all_predictions = validate(model, val_loader, criterion, device)

print(f"Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}")
print(f"Overall Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_names))

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions, cmap=plt.cm.magma
                                      , display_labels=class_names, ax=ax)
ax.set_title("Confusion Matrix")
plt.show()