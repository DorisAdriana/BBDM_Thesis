import os
import numpy as np
from PIL import Image

def add_gaussian_noise(image):
    """Add Gaussian noise to an image."""
    row, col = image.size
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (col, row))
    gauss = gauss.reshape(col, row)
    noisy = np.array(image) + gauss
    noisy = np.clip(noisy, 0, 255)
    return Image.fromarray(noisy.astype('uint8'))

def process_images(input_folder, output_folder):
    """Process images from input_folder, save to output_folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_noisy = add_gaussian_noise(img)
            img_noisy.save(os.path.join(output_folder, filename))

# Set your input and output folder paths
input_folder = '/Users/dorisadriana/Downloads/Celebdata'
output_folder = '/Users/dorisadriana/Downloads/Output'

process_images(input_folder, output_folder)
