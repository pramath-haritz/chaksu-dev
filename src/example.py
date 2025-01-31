import os
from chaksu.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

image_path = os.path.join(BASE_DIR, 'images', '4.jpg')
# read image as numpy array, or base64, or bytes...
with open(image_path, 'rb') as f:
    image_data = f.read()

# Segment with vessel model
vessel_img, vessel_logs = Pipeline(image_data, method="vessel")
print(vessel_logs)

# Segment with OD/OC model
odoc_img, odoc_logs = Pipeline(image_data, method="odoc")
print(odoc_logs)

# Run general diagnosis
# Segment with OD/OC model
logs = Pipeline(image_data, method="general")
print(logs)