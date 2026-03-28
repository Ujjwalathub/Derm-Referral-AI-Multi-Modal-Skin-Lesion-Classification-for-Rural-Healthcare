import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import h5py
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class ISICDataset(Dataset):
    """
    ISIC 2024 Dataset for skin lesion classification.
    Supports both HDF5 and raw image folder formats.
    """
    def __init__(self, metadata_path, clinical_features, categorical_features, 
                 image_size=224, is_train=True, hdf5_path=None, image_dir=None):
        # Fixed: DtypeWarning for mixed types in metadata
        self.metadata = pd.read_csv(metadata_path, low_memory=False)
        self.clinical_features = clinical_features
        self.categorical_features = categorical_features
        self.image_size = image_size
        self.is_train = is_train
        self.hdf5_path = hdf5_path
        self.image_dir = image_dir
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.encoders = {}
        
        # Prepare features
        self.features = self._prepare_features()
        
        # Don't open HDF5 file in __init__ to avoid pickling issues with multiprocessing
        self.hdf5_file = None
    
    def _prepare_features(self):
        """Preprocess clinical and categorical features"""
        feature_list = []
        
        # Clinical features (numerical)
        if self.clinical_features:
            clinical_data = self.metadata[self.clinical_features].fillna(0).values
            if self.is_train:
                clinical_data = self.scaler.fit_transform(clinical_data)
            else:
                clinical_data = self.scaler.transform(clinical_data)
            feature_list.append(clinical_data)
        
        # Categorical features
        for cat_feat in self.categorical_features:
            if self.is_train:
                self.encoders[cat_feat] = LabelEncoder()
                encoded = self.encoders[cat_feat].fit_transform(
                    self.metadata[cat_feat].fillna('unknown')
                )
            else:
                encoded = self.encoders[cat_feat].transform(
                    self.metadata[cat_feat].fillna('unknown')
                )
            feature_list.append(encoded.reshape(-1, 1))
        
        # Concatenate all features
        if feature_list:
            return np.hstack(feature_list).astype(np.float32)
        else:
            return np.zeros((len(self.metadata), 1), dtype=np.float32)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get image ID and target
        isic_id = self.metadata.iloc[idx]['isic_id']
        target = self.metadata.iloc[idx]['target']
        
        # Load image
        if self.image_dir:
            # Load from folder
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            image = Image.open(img_path).convert('RGB')
        elif self.hdf5_path:
            # Open HDF5 file on-demand to avoid pickling issues
            if self.hdf5_file is None:
                self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            
            # Access the specific image by isic_id
            raw_data = self.hdf5_file[isic_id][()]
            
            # FIX: Ensure we are decoding the bytes if they are stored as a blob
            import io
            
            try:
                # Try to open as a byte stream (standard for JPEG blobs in HDF5)
                image = Image.open(io.BytesIO(raw_data)).convert('RGB')
            except:
                # Fallback if it is raw pixel data
                image = Image.fromarray(raw_data.astype('uint8')).convert('RGB')
        else:
            raise ValueError("No image source specified (hdf5_path or image_dir)")
        
        # Resize and normalize
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get metadata features
        metadata_features = torch.from_numpy(self.features[idx])
        
        # Target
        target = torch.tensor(target, dtype=torch.float32)
        
        return image, metadata_features, target
    
    def __del__(self):
        # Close HDF5 file when done
        if self.hdf5_file:
            self.hdf5_file.close()
