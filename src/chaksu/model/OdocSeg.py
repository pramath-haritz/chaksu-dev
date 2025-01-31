"""
Implements the YOLO-based OD/OC segmentation.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from .BaseModel import BaseModel
from ..utils.io import preprocess_image_data

from pathlib import Path
# Constants for the base directory and model weights directory
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust the base directory as needed
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"

retina_segmentation_model_path = MODEL_WEIGHTS_DIR/"retina_seg.pt"

class RetinaSegmentationModel(BaseModel):
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
            super().__init__("Retina Segmentation")
            self.model = YOLO(path)
            self.initialized = True

    def flatten_points(self, points):
        return [(point[0][0], point[0][1]) for point in points]

    def calculate_distances(self, tuple1, tuple2):
        distances = []
        for point1 in tuple1:
            for point2 in tuple2:
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append(distance)
        min_distance = min(distances)
        max_distance = max(distances)
        return min_distance, max_distance

    def process_image(self, image_data, is_numpy=False):
        image_np = preprocess_image_data(image_data)  # default 512,512
        results = self.model.predict(image_np, retina_masks=True, show_boxes=False)
        logs = []
        processed_image = image_np.copy()

        if len(results) == 0:
            logs.append("No optic structures detected")
            return processed_image, "\n".join(logs)

        for r in results:
            try:
                flat_disc_contour = r.masks.xy[0]
                disc_contour = np.array(flat_disc_contour).reshape((-1, 1, 2)).astype(np.int32)
                flat_cup_contour = r.masks.xy[1]
                cup_contour = np.array(flat_cup_contour).reshape((-1, 1, 2)).astype(np.int32)

                _, disc_max = self.calculate_distances(flat_disc_contour, flat_disc_contour)
                _, cup_max = self.calculate_distances(flat_cup_contour, flat_cup_contour)
                disc_cup_min, _ = self.calculate_distances(flat_disc_contour, flat_cup_contour)

                logs.append(f"Disc max: {disc_max:.2f}, Cup max: {cup_max:.2f}, Rim min: {disc_cup_min:.2f}")
                logs.append(f"CDR: {cup_max/disc_max:.2f}, RDR: {disc_cup_min/disc_max:.2f}")

                processed_image = cv2.drawContours(processed_image, [disc_contour, cup_contour], -1, (0, 0, 255), 10)
            except IndexError as e:
                logs.append(f"IndexError: {str(e)}")

        return processed_image, "\n".join(logs)
