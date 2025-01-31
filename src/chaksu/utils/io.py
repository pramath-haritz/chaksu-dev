"""
Handles reading images in various formats, e.g. NumPy arrays, base64 strings, or raw bytes.
"""
import base64
import numpy as np
from io import BytesIO
from PIL import Image


def convert_to_pil(image_data):
    """
    Convert an input (NumPy array, base64 string, or raw bytes) to a PIL Image.
    """
    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data).convert('RGB')
    elif isinstance(image_data, str):  # base64
        image_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(image_bytes)).convert('RGB')
    elif isinstance(image_data, bytes):  # raw binary
        return Image.open(BytesIO(image_data)).convert('RGB')
    else:
        raise ValueError("Unsupported image format")


def center_crop_and_resize(pil_image, target_size=(512, 512)):
    """
    Crops an image to preserve aspect ratio, then resizes.
    """
    width, height = pil_image.size
    target_ratio = target_size[0] / target_size[1]
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Wider
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = height
    else:
        # Taller
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
        right = width
        bottom = top + new_height

    cropped = pil_image.crop((left, top, right, bottom))
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return resized


def preprocess_image_data(image_data, target_size=(512, 512)):
    """
    Convenience function that uses convert_to_pil and center_crop_and_resize to produce
    a NumPy array of shape [H,W,3].
    """
    pil_img = convert_to_pil(image_data)
    resized = center_crop_and_resize(pil_img, target_size)
    return np.array(resized)
