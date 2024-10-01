import cv2
import numpy as np
import os
from skimage import io

def apply_clahe(image):
    # Convert to grayscale if not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the grayscale image
    return clahe.apply(gray_image)

# Example for applying CLAHE to the dataset
def preprocess_images(image_dir, output_dir):
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = io.imread(image_path)
        preprocessed_image = apply_clahe(image)
        output_path = os.path.join(output_dir, filename)
        io.imsave(output_path, preprocessed_image)

preprocess_images('dataset/images', 'dataset/preprocessed')
