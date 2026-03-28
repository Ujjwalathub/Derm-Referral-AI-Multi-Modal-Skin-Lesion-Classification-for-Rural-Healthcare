# Configuration for RTX 3050 (6GB VRAM)
import torch

# Paths - Updated to new data directory
# Images are in E:\IIT Ropar\Data, metadata CSVs are in SLICE-3D folder
IMAGE_DATA_DIR = r"E:\IIT Ropar\Data"
METADATA_DIR = "SLICE-3D (ISIC 2024)"

TRAIN_HDF5 = f"{METADATA_DIR}/train-image.hdf5"
TRAIN_METADATA = f"{METADATA_DIR}/train-metadata.csv"
TEST_HDF5 = f"{METADATA_DIR}/test-image.hdf5"
TEST_METADATA = f"{METADATA_DIR}/test-metadata.csv"

# Optional raw image folder support (only use if all images are present)
# The Data folder only has 4,849 images, but metadata has 401,059 entries
# Use HDF5 instead which contains the full dataset
TRAIN_IMAGE_DIR = None  # Set to None to force HDF5 usage

# Model settings
IMAGE_SIZE = 224  # EfficientNet-B0 native resolution
BATCH_SIZE = 16  # Optimized for RTX 3050
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# GPU optimization for RTX 3050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # Mixed precision for 50% speedup

# Hardware-specific optimizations for Ryzen 5 7000HS & RTX 3050
torch.backends.cudnn.benchmark = True  # Optimizes CNN algorithms for RTX 3050

# Clinical features to use (as per documentation requirements)
CLINICAL_FEATURES = [
    'tbp_lv_area_perim_ratio',  # Border jaggedness
    'tbp_lv_symm_2axis'  # Asymmetry
]

CATEGORICAL_FEATURES = []  # Simplified for non-specialist use

# Class imbalance handling (will be calculated from data)
POS_WEIGHT = None  # Set during training
