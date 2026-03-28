import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import os   

from config import *
from dataset import ISICDataset
from model import HybridMultiModalModel

def calculate_pos_weight(metadata_path):
    """Calculate class imbalance weight for BCEWithLogitsLoss"""
    df = pd.read_csv(metadata_path, low_memory=False)  # Fixed: DtypeWarning for mixed types
    benign = (df['target'] == 0).sum()
    malignant = (df['target'] == 1).sum()
    return benign / malignant

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, metadata, targets in pbar:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=USE_AMP):
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, metadata, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.to(device)
            
            with autocast(enabled=USE_AMP):
                outputs = model(images, metadata)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(loader), correct / total

def main():
    print(f"Using device: {DEVICE}")
    
    # Calculate class imbalance weight
    pos_weight = calculate_pos_weight(TRAIN_METADATA)
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Load dataset
    print("Loading dataset...")
    train_image_dir = TRAIN_IMAGE_DIR if TRAIN_IMAGE_DIR and os.path.isdir(TRAIN_IMAGE_DIR) else None
    train_hdf5 = TRAIN_HDF5 if os.path.isfile(TRAIN_HDF5) else None
    if not train_image_dir and not train_hdf5:
        raise FileNotFoundError(
            f"Could not find training images. Expected folder '{TRAIN_IMAGE_DIR}' or file '{TRAIN_HDF5}'."
        )

    full_dataset = ISICDataset(
        TRAIN_METADATA,
        CLINICAL_FEATURES,
        CATEGORICAL_FEATURES,
        IMAGE_SIZE,
        is_train=True,
        hdf5_path=train_hdf5,
        image_dir=train_image_dir,
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Share preprocessors with validation set
    val_dataset.dataset.is_train = False
    
    # DataLoaders with hardware optimizations
    # num_workers=0 to avoid h5py pickling error on Windows (multiprocessing spawn issue)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    num_features = full_dataset.features.shape[1]
    model = HybridMultiModalModel(num_features).to(DEVICE)
    print(f"Model created with {num_features} metadata features")
    
    # Loss with class imbalance handling
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=USE_AMP)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'scaler': scaler.state_dict(),
                'preprocessors': {
                    'scaler': full_dataset.scaler,
                    'encoders': full_dataset.encoders
                }
            }, 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
