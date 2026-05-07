import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CRITICAL PATH FIX: Robustly find project root to import backend
# -----------------------------------------------------------------------------
# Get current working directory
current_dir = Path(os.getcwd())

# Logic: If we are in 'notebooks', go up. If we are at root, stay.
if current_dir.name == 'notebooks':
    project_root = current_dir.parent
elif (current_dir / 'backend').exists():
    project_root = current_dir
else:
    # Fallback
    project_root = current_dir.parent

# Add to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"✅ Added {project_root} to sys.path")

from backend.model import OsteoClassifier
from backend.data_pipeline import apply_clahe
from config.settings import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

BATCH_SIZE = 16
EPOCHS = 15 # Reduced for testing/running
LEARNING_RATE = 0.0001
DATA_DIR = project_root / "data/raw"

class CLAHETransform:
    """Wrapper to apply CLAHE to PIL images in the transform pipeline"""
    def __call__(self, img):
        # apply_clahe returns 'L' mode image
        enhanced = apply_clahe(img)
        # Convert to RGB for DenseNet
        return enhanced.convert("RGB")

# Training Transforms (Augmentation + Preprocessing)
train_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# Validation Transforms (No Augmentation, Just Preprocessing)
val_transforms = transforms.Compose([
    CLAHETransform(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# Check if data directories exist
if not (DATA_DIR / 'train').exists() or not (DATA_DIR / 'val').exists():
    print(f"❌ Data directories not found at {DATA_DIR}")
    print("Please ensure 'data/raw/train' and 'data/raw/val' exist.")
    sys.exit(1)

train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training Samples: {len(train_dataset)}")
print(f"Validation Samples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

model = OsteoClassifier(num_classes=5)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("Starting Training...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-" * 10)

    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc.item())

    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)

    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc.item())

    print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

    # --- SAVE BEST MODEL ---
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        # Ensure the directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ New Best Model Saved to {MODEL_PATH}! (Acc: {best_acc:.4f})")

print("Training Complete.")
