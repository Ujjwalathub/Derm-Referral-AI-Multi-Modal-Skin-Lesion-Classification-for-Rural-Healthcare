"""
Setup script for AI4Healthcare project
Run this to verify your environment is ready for training
"""

import sys
import subprocess

def check_python_version():
    """Verify Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Verify CUDA-enabled PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available. Install with:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Verify all required packages"""
    required = ['pandas', 'numpy', 'h5py', 'PIL', 'sklearn', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_dataset():
    """Verify dataset files exist"""
    import os
    
    required_files = [
        'SLICE-3D (ISIC 2024)/train-image.hdf5',
        'SLICE-3D (ISIC 2024)/train-metadata.csv',
        'SLICE-3D (ISIC 2024)/test-image.hdf5',
        'SLICE-3D (ISIC 2024)/test-metadata.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1e9
            print(f"✓ {file} ({size:.1f} GB)")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("AI4Healthcare Environment Check")
    print("=" * 60)
    
    print("\n1. Python Version:")
    py_ok = check_python_version()
    
    print("\n2. CUDA & PyTorch:")
    cuda_ok = check_cuda()
    
    print("\n3. Dependencies:")
    deps_ok = check_dependencies()
    
    print("\n4. Dataset Files:")
    data_ok = check_dataset()
    
    print("\n" + "=" * 60)
    if py_ok and cuda_ok and deps_ok and data_ok:
        print("✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python train.py      # Train the model")
        print("  2. python inference.py  # Run inference demo")
    else:
        print("❌ Some checks failed. Fix the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
