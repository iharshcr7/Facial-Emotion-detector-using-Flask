# =====================================================
# RAF-DB Facial Emotion Recognition - EfficientNet-B0 CPU Optimized
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
from tqdm import tqdm
import os

# ------------------------------
# 1️⃣ Device setup
# ------------------------------
device = torch.device("cpu")  # force CPU
print("Using device:", device)

# ------------------------------
# 2️⃣ Emotion mapping
# ------------------------------
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# ------------------------------
# 3️⃣ Transforms (smaller image size)
# ------------------------------
train_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------------
# 4️⃣ Load datasets
# ------------------------------
train_dir = r"C:\Users\harsh\Downloads\archive\DATASET\train"
test_dir  = r"C:\Users\harsh\Downloads\archive\DATASET\test"

full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Split train into train + validation
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # smaller batch
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ------------------------------
# 5️⃣ Quick sanity checks
# ------------------------------
print("Class to Index Mapping:", full_train_dataset.class_to_idx)
print("Training set distribution:", Counter([label for _, label in full_train_dataset.samples]))
print("Validation size:", len(val_dataset), "Train size:", len(train_dataset))

# ------------------------------
# 6️⃣ Load pretrained EfficientNet-B0 and freeze backbone
# ------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
model = model.to(device)

# ------------------------------
# 7️⃣ Loss, optimizer, scheduler
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)  # only classifier
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ------------------------------
# 8️⃣ Early stopping setup
# ------------------------------
early_stop_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
num_epochs = 15
best_model_path = "best_efficientnet_rafdb_cpu.pth"

# ------------------------------
# 9️⃣ Training loop with validation, progress, early stopping
# ------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training Batches")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # ---------------- Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Learning rate scheduler step
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation improved. Best model saved to {best_model_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# ------------------------------
# 10️⃣ Test evaluation using best model
# ------------------------------
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(best_model_path))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Batches"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\nTest Accuracy: {test_acc:.2f}%")

# ------------------------------
# 11️⃣ Example prediction mapping
# ------------------------------
example_label = 1
print(f"Numeric label {example_label} => Emotion: {emotion_map[example_label]}")
