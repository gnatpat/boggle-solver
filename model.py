import os
import json
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
        self.letter2idx = {chr(i + 65): i for i in range(26)}
        self.idx2letter = {i: chr(i + 65) for i in range(26)}

        self.img_ids = list(self.labels.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_id = self.img_ids[idx]
        img_name = f'{image_id}_processed.png'
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("L")  # grayscale

        label = self.letter2idx[self.labels[image_id]]

        if self.transform:
            image = self.transform(image)

        return image, label


class BoggleCNN(nn.Module):
    def __init__(self):
        super(BoggleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*8*8, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class RandomRotation90:
    def __call__(self, img: Image.Image):
        angles = [0, 90, 180, 270]
        jitter = random.uniform(-5, 5)
        angle = random.choice(angles)
        return img.rotate(angle)

def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        RandomRotation90(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor()
    ])

    dataset = BoggleDataset("images", "labels.json", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoggleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------
    # 4. Training Loop
    # ------------------------------
    num_epochs = 30
    best_val_acc = 0
    patience = 5   # how many epochs to wait before stopping
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

    dummy_input = torch.randn(1, 1, 32, 32, device=device)
    torch.onnx.export(model, (dummy_input,), "boggle_cnn.onnx",
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    opset_version=11)
    
if __name__=="__main__":
    main()