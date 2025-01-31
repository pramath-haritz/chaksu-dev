from __future__ import annotations

import logging
from pathlib import Path
import googledrivedownloader as gdd

# Constants for the base directory and model weights directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust the base directory as needed
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"

# List of files to download with Google Drive IDs and destination file names
FILES_TO_DOWNLOAD = [
    {
        "file_id": "1qaOIymD6JalAAxDq-FFVzHQcbsI6qv2w",  # Replace with the actual file ID
        "file_name": "retina_seg.pt",
    },
    {
        "file_id": "1tGbmeM5eRbaupPG7v2oxKEYNhEi2PbUc",  # Replace with the actual file ID
        "file_name": "vessel_seg.pth",
    },
    {
        "file_id": "1B2cyfFpd6Z4vtva-qH-p_g4rynC1K5-j",  # Replace with the actual file ID
        "file_name": "flair_resnet.pth",
    },
]


def download_model_weights():
    """
    Create the model_weights directory and download model files from Google Drive.

    Raises:
        ValueError: If the download fails for any file.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Ensure the model_weights directory exists
    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model weights directory created at: {MODEL_WEIGHTS_DIR}")

    # Download each file in the list
    for file_info in FILES_TO_DOWNLOAD:
        try:
            dest_path = MODEL_WEIGHTS_DIR / file_info["file_name"]
            gdd.download_file_from_google_drive(
                file_id=file_info["file_id"],
                dest_path=str(dest_path),  # Convert Path to str for compatibility
                unzip=False,  # Set to True if you expect zipped files
            )
            logger.info(f"Downloaded {file_info['file_name']} successfully.")
        except Exception as e:
            logger.error(f"Failed to download {file_info['file_name']}: {e}")
            raise ValueError(f"Error downloading {file_info['file_name']}.") from e

if __name__ == "__main__":
    download_model_weights()