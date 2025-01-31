"""
Implements the Retinal Vessel Segmentation using UNet.
"""

import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from .BaseModel import BaseModel
from ..utils.io import preprocess_image_data

from pathlib import Path
# Constants for the base directory and model weights directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust the base directory as needed
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path to the vessel segmentation model
ratina_vessel_state_dict = MODEL_WEIGHTS_DIR / "vessel_seg.pth"


class RetinalVesselSegmentation(BaseModel):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"Creating new instance of {cls.__name__}")
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, path=None):
        if not self.initialized:
            print(f"Initializing {self.__class__.__name__}")
            super().__init__("Retinal Vessel Segmentation")
            self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
            self.model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.initialized = True

    def process_image(self, image_data, is_numpy=False):
        # Preprocess
        image_np = preprocess_image_data(image_data, target_size=(512, 512))
        original_size = (image_np.shape[1], image_np.shape[0])  # (width, height)
        input_image = self.transform(Image.fromarray(image_np)).unsqueeze(0)
        input_image = input_image.to(DEVICE)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_image)
            output = torch.sigmoid(output).squeeze()
            output = output.cpu().numpy()

        # Resize back to original if needed
        if output.shape != (original_size[1], original_size[0]):
            output = cv2.resize(output, original_size)

        # Threshold
        binary_mask = output > 0.5
        segmented_image = image_np.copy()
        segmented_image[binary_mask] = [0, 0, 255]
        segmented_image = segmented_image.astype(np.uint8)

        # Calculate vessel density
        vessel_density = np.mean(binary_mask) * 100
        logs = f"Vessel Density: {vessel_density:.2f}%"
        return segmented_image, logs
