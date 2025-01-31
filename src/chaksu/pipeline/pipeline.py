
"""
Pipeline function for performing segmentation.
"""
from ..model.RetinaSeg import RetinalVesselSegmentation
from ..model.OdocSeg import RetinaSegmentationModel
from ..model.Flair import GeneralPathologyModel
from ..visulization import show_image

def Pipeline(image_data,path="model path missing", method="odoc"):
    """
    Accepts an image_data and segments it using either odoc-seg or vessel-seg.
    Returns the segmented image and logs.
    """
    if method == "odoc":
        model = RetinaSegmentationModel(path)
        segmented_image, logs = model.process_image(image_data)
        show_image(segmented_image)
        return segmented_image, logs
    elif method == "vessel":
        model = RetinalVesselSegmentation(path)
        segmented_image, logs = model.process_image(image_data)
        show_image(segmented_image)
        return segmented_image, logs
    elif method == "general":
        model = GeneralPathologyModel(path)
        logs = model.process_image(image_data)
        return logs
    else:
        raise ValueError("Unknown method, use either 'odoc' or 'vessel' or 'general'.")
