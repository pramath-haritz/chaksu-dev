"""
Abstract base class for a model.
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def process_image(self, image_data, is_numpy=False):
        pass
