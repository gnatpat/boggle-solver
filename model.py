import os
import json
import random
from PIL import Image
from attr import frozen
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

INPUT_SIZE = 32
NUM_CLASSES = 31  # 26 letters + 5 extra (An, Er, He, In, Th)

def get_index_to_letter_maps():
    base_letters = [chr(i + 65) for i in range(26)]  # A-Z
    extra_letters = ['AN', 'ER', 'HE', 'IN', 'TH']
    all_letters = base_letters + extra_letters
    index2letter = {i: letter for i, letter in enumerate(all_letters)}
    letter2index = {letter.upper(): i for i, letter in enumerate(all_letters)}
    return letter2index, index2letter


@frozen
class BoggleCNNConfig:
    num_classes: int = NUM_CLASSES
    conv1_out_channels: int = 32
    conv1_kernel_size: int = 3
    conv2_out_channels: int = 128
    conv2_kernel_size: int = 3
    pool_kernel_size: int = 2
    fc1_out_features: int = 128

@frozen
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 30
    patience: int = 5
    val_split: float = 0.2

@frozen
class Config:
    boggle_cnn: BoggleCNNConfig = BoggleCNNConfig()
    training: TrainingConfig = TrainingConfig()
    img_folder: str = "images"
    labels_file: str = "labels.json"

# ------------------------------
# 1. Custom Dataset
# ------------------------------
class BoggleDataset(Dataset):
    def __init__(self, img_folder, labels_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform

        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        # Map letters to indices 0-25
        self.letter2idx, self.idx2letter = get_index_to_letter_maps()

        self.img_ids = list(self.labels.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]
        img_name = f'{image_id}_processed.png'
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("L")  # grayscale

        label = self.letter2idx[self.labels[image_id].upper()]

        if self.transform:
            image = self.transform(image)

        return image, label

class BoggleCNN(nn.Module):
    def __init__(self, config):
        super(BoggleCNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, config.conv1_out_channels, config.conv1_kernel_size, padding=1)
        self.conv2 = nn.Conv2d(config.conv1_out_channels, config.conv2_out_channels, config.conv2_kernel_size, padding=1)
        self.pool = nn.MaxPool2d(config.pool_kernel_size, config.pool_kernel_size)
        pooled_size = INPUT_SIZE // (config.pool_kernel_size ** 2)
        self.cnn_flattened_size = config.conv2_out_channels * pooled_size * pooled_size
        self.fc1 = nn.Linear(self.cnn_flattened_size, config.fc1_out_features)
        self.fc2 = nn.Linear(config.fc1_out_features, config.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.cnn_flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class RandomRotation90:
    def __call__(self, img: Image.Image):
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        return img.rotate(angle)

def main(config: Config):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        RandomRotation90(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor()
    ])

    dataset = BoggleDataset(config.img_folder, config.labels_file, transform=transform)

    train_size = int((1 - config.training.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoggleCNN(config.boggle_cnn).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # ------------------------------
    # 4. Training Loop
    # ------------------------------
    num_epochs = config.training.num_epochs
    patience = config.training.patience
    best_val_acc = 0
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    # ------------------------------
    # 5. Save Model & Export ONNX
    # ------------------------------
    torch.save(model.state_dict(), "boggle_cnn.pth")

    dummy_input = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, device=device)
    torch.onnx.export(model, (dummy_input,), "boggle_cnn.onnx",
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    opset_version=11)
    
if __name__=="__main__":
    main(config=Config())