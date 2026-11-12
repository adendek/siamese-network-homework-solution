import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://github.com/adendek/siamese-network-homework-solution/releases/download/data-v1.0/faces.zip"
DEST_ZIP = Path("data/faces.zip")
EXTRACT_DIR = Path("data/")


def setup_dataset():
    """Download and extract dataset for the release."""
    if Path("data/split/train").exists():
        print("Dataset already unpacked.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEST_ZIP.exists():
        print(f"Downloading faces.zip from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DEST_ZIP)
        print("Download complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(DEST_ZIP, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Dataset ready in data/")


if __name__ == "__main__":
    setup_dataset()
