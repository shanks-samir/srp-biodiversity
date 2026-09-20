import os
import sys
import zipfile

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Potsdam")

def check_files():
    print(f"Checking dataset directory at: {os.path.abspath(DATA_DIR)}")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return False

    files = os.listdir(DATA_DIR)
    print(f"Found {len(files)} items in {DATA_DIR}:")
    for f in sorted(files):
        fpath = os.path.join(DATA_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")

    required = ["4_Ortho_RGBIR.zip", "5_Labels_all.zip"]
    missing = [req for req in required if req not in files]
    if missing:
        print(f"\nWarning: Missing required files for TorchGeo: {missing}")
        return False

    print("\nAll required archives (4_Ortho_RGBIR.zip and 5_Labels_all.zip) are present!")
    return True

def test_loader():
    try:
        import torchgeo
        from datasets.potsdam import TorchGeoPotsdamDataset
        print("\ntorchgeo is installed! Testing TorchGeoPotsdamDataset initialization...")
        ds = TorchGeoPotsdamDataset(root=DATA_DIR, split="train", crop_size=512)
        print(f"Successfully loaded dataset! Total samples: {len(ds)}")
        sample = ds[0]
        print(f"Sample 0 Image Shape (RGB): {sample['image'].shape} (min: {sample['image'].min():.2f}, max: {sample['image'].max():.2f})")
        print(f"Sample 0 NDVI Shape: {sample['ndvi'].shape} (min: {sample['ndvi'].min():.2f}, max: {sample['ndvi'].max():.2f})")
        print(f"Sample 0 Shannon Shape: {sample['shannon'].shape} (min: {sample['shannon'].min():.2f}, max: {sample['shannon'].max():.2f})")
        print(f"Sample 0 Mask Shape: {sample['mask'].shape} (classes present: {sample['mask'].unique().tolist()})")
        print("\n--> Dataset verified and ready for training!")
    except ImportError:
        print("\nNote: torchgeo is not installed in the current environment.")
        print("To install all dependencies before running training, run:")
        print("    pip install -r requirements.txt")

if __name__ == "__main__":
    if check_files():
        test_loader()
