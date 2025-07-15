#!/usr/bin/env python3
"""
Script to download the DataCo Smart Supply Chain dataset from Kaggle.
Requires kaggle API credentials to be set up (~/.kaggle/kaggle.json).
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def check_kaggle_credentials():
    """Check if Kaggle API credentials exist."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Error: Kaggle API credentials not found!")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

def install_kaggle():
    """Install kaggle package if not already installed."""
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def download_dataset():
    """Download the DataCo Supply Chain dataset."""
    dataset = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
    output_dir = "data"
    
    print(f"Downloading dataset from Kaggle...")
    subprocess.check_call(["kaggle", "datasets", "download", "-d", dataset, "-p", output_dir])
    
    # Unzip the dataset
    zip_path = os.path.join(output_dir, "dataco-smart-supply-chain-for-big-data-analysis.zip")
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Error: Download failed!")
        sys.exit(1)

def main():
    """Main function to orchestrate dataset download."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check for Kaggle credentials
    check_kaggle_credentials()
    
    # Install kaggle package if needed
    install_kaggle()
    
    # Download and extract the dataset
    download_dataset()

if __name__ == "__main__":
    main() 