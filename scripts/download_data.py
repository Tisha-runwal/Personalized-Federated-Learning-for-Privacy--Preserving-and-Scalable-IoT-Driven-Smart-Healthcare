"""Download datasets for PFL-HCare experiments."""
import os, zipfile, urllib.request, sys

HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_har(root: str = "./datasets") -> str:
    har_dir = os.path.join(root, "UCI HAR Dataset")
    if os.path.exists(har_dir):
        print(f"UCI HAR already exists at {har_dir}")
        return har_dir
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "har.zip")
    print("Downloading UCI HAR Dataset...")
    urllib.request.urlretrieve(HAR_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)
    print(f"UCI HAR Dataset ready at {har_dir}")
    return har_dir

def download_heart_disease(root: str = "./datasets") -> str:
    heart_path = os.path.join(root, "heart_disease", "processed.cleveland.data")
    if os.path.exists(heart_path):
        print(f"Heart Disease already exists at {heart_path}")
        return heart_path
    os.makedirs(os.path.join(root, "heart_disease"), exist_ok=True)
    print("Downloading Cleveland Heart Disease Dataset...")
    urllib.request.urlretrieve(HEART_URL, heart_path)
    print(f"Heart Disease Dataset ready at {heart_path}")
    return heart_path

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    root = sys.argv[2] if len(sys.argv) > 2 else "./datasets"
    if dataset in ("har", "all"):
        download_har(root)
    if dataset in ("heart", "all"):
        download_heart_disease(root)
    print("Done.")
