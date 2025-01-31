"""
Implements the FLAIR-based general pathology model.
"""

import torch
import numpy as np
from .BaseModel import BaseModel
from flair import FLAIRModel
from ..utils.io import preprocess_image_data
from pathlib import Path
# Constants for the base directory and model weights directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust the base directory as needed
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"

GeneralPathologyModel_path = MODEL_WEIGHTS_DIR / "flair_resnet.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeneralPathologyModel(BaseModel):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"Creating new instance of {cls.__name__}")
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, path = None):
        if not self.initialized:
            print(f"Initializing {self.__class__.__name__}")
            super().__init__("General Pathology")
            self.model = FLAIRModel(from_checkpoint=True, weights_path=path)
            self.text_categories = [
                "Normal","Age-Related Macular Degeneration", "Macular Edema", "Diabetic Retinopathy",
                "Glaucoma","Cataract","Retinal Vein Occlusion","Lesion in the Macula", "Retinal Detachment","Hypertensive Retinopathy"
            ]
            self.initialized = True

    def process_image(self, image_data, is_numpy=False, ai_assistant_mode=True, requested_entities='ALL'):
        image_np = preprocess_image_data(image_data, target_size=(512, 512))
        if ai_assistant_mode:
            if requested_entities == 'ALL':
                categories = self.text_categories
            else:
                # Example: user can pass a list of entities.
                # We'll just do a minimal example.
                categories = ['Normal']
                if requested_entities:
                    for entity in requested_entities:
                        categories.append(str(entity))
        else:
            categories = self.text_categories

        probs, logits = self.model(image_np, categories)
        probs = probs.squeeze()
        logs = []
        for category, prob in zip(categories, probs):
            logs.append(f"{category}: {prob*100:.2f}%")
        return None, "\n".join(logs)