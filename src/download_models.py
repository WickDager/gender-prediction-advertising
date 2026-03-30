#!/usr/bin/env python3
"""
Download models from Hugging Face Hub.
Usage: python download_models.py
"""

import os
from huggingface_hub import hf_hub_download, login
from pathlib import Path

# ================= CONFIGURATION =================
REPO_ID = "YOUR_USERNAME/gender-prediction-lightgbm"   # Replace with your HF repo ID
FILES_TO_DOWNLOAD = ["gender_classifier.pkl", "imputer.pkl"]
LOCAL_DIR = "models"                                    # Where to save the files
USE_AUTH = False                                        # Set True if repo is private
# ==================================================

def download_models():
    # Create local directory if it doesn't exist
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    # Optional: login if required (for private repos)
    if USE_AUTH:
        print("Hugging Face login required for private repo.")
        login()  # Will prompt for token if not already logged in

    for filename in FILES_TO_DOWNLOAD:
        print(f"Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,  # Copy file, not symlink
                resume=True
            )
            print(f"Saved to: {downloaded_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            print("Make sure the file exists in the HF repo and the repo ID is correct.")

if __name__ == "__main__":
    download_models()