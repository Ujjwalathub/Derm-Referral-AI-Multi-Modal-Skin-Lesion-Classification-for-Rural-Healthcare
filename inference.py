import torch
import h5py
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from config import *
from model import HybridMultiModalModel

class ReferralSystem:
    """Convert malignancy probability to tri-color referral action"""
    
    @staticmethod
    def get_action(probability):
        if probability < 0.2:
            return "GREEN", "No Referral Needed (Routine Monitoring)"
        elif probability <= 0.5:
            return "YELLOW", "Referral Suggested (Tele-consult with Specialist)"
        else:
            return "RED", "URGENT Referral (Immediate Physical Exam)"

def load_model(checkpoint_path, num_features):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = HybridMultiModalModel(num_features).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['preprocessors']

def _load_image_from_sources(*, isic_id: str, image_dir: str | None, hdf5_path: str | None) -> Image.Image:
    if image_dir:
        img_path = os.path.join(image_dir, f"{isic_id}.jpg")
        return Image.open(img_path).convert("RGB")
    if hdf5_path:
        with h5py.File(hdf5_path, "r") as f:
            img_array = f[isic_id][()]
        return Image.fromarray(img_array).convert("RGB")
    raise ValueError("Either image_dir or hdf5_path must be provided.")

def predict_single(model, *, hdf5_path=None, image_dir=None, isic_id=None, metadata_row=None, preprocessors=None):
    """Predict malignancy probability for a single image"""
    
    img = _load_image_from_sources(isic_id=isic_id, image_dir=image_dir, hdf5_path=hdf5_path)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Prepare metadata
    clinical_data = metadata_row[CLINICAL_FEATURES].fillna(0).values.reshape(1, -1)
    clinical_scaled = preprocessors['scaler'].transform(clinical_data)
    
    categorical_encoded = []
    for col in CATEGORICAL_FEATURES:
        encoded = preprocessors['encoders'][col].transform([metadata_row[col] if pd.notna(metadata_row[col]) else 'unknown'])
        n_classes = len(preprocessors['encoders'][col].classes_)
        one_hot = np.eye(n_classes)[encoded]
        categorical_encoded.append(one_hot)
    
    if categorical_encoded:
        categorical_encoded = np.concatenate(categorical_encoded, axis=1)
    else:
        categorical_encoded = np.zeros((1, 0))
    
    features = np.concatenate([clinical_scaled, categorical_encoded], axis=1)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(img_tensor, features_tensor)
            probability = torch.sigmoid(logits).item()
    
    return probability

def demo():
    """Demo inference with referral action"""
    print("Loading model...")
    
    # Load test metadata to get feature count
    test_metadata = pd.read_csv(TEST_METADATA)
    
    # Load checkpoint to get preprocessors
    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    preprocessors = checkpoint['preprocessors']
    
    # Calculate feature dimensions
    num_clinical = len(CLINICAL_FEATURES)
    num_categorical = sum(len(preprocessors['encoders'][col].classes_) for col in CATEGORICAL_FEATURES)
    num_features = num_clinical + num_categorical
    
    model, preprocessors = load_model('best_model.pth', num_features)
    
    # Test on first sample
    sample_row = test_metadata.iloc[0]
    isic_id = sample_row['isic_id']
    
    print(f"\nAnalyzing: {isic_id}")
    probability = predict_single(model, hdf5_path=TEST_HDF5, isic_id=isic_id, metadata_row=sample_row, preprocessors=preprocessors)
    
    color, action = ReferralSystem.get_action(probability)
    
    print(f"Malignancy Probability: {probability:.2%}")
    print(f"Referral Action: [{color}] {action}")

if __name__ == "__main__":
    demo()
