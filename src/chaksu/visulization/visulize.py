
"""
Functions to display images and create overlays.
"""
import matplotlib.pyplot as plt
import numpy as np


def show_image(image_np, title="Image"):
    plt.figure(figsize=(8,8))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_overlay(original_np, mask_np, overlay_color=(255,0,0), alpha=0.3, title="Overlay"):
    """
    Overlay a mask onto an image. The mask is assumed to be boolean or 0/1
    """
    overlay_image = original_np
    overlay_image[mask_np > 0] = overlay_color
    # For alpha blending, if desired.
    blended = (original_np * (1 - alpha) + overlay_image * alpha).astype(np.uint8)
    plt.figure(figsize=(8,8))
    plt.imshow(blended)
    plt.title(title)
    plt.axis('off')
    plt.show()
