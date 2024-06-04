from PIL import Image, ImageEnhance
import numpy as np

def improve_mri_contrast(input_path, export_path):
    # Open an image file
    with Image.open(input_path) as img:
        # Ensure the image is in grayscale mode
        gray_img = img.convert('L')

        # Convert the grayscale image to a numpy array
        img_array = np.array(gray_img)

        # Improve the contrast by stretching the pixel values
        # Apply contrast stretching
        p2, p98 = np.percentile(img_array, (2, 98))
        img_rescale = np.clip((img_array - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)

        # Convert back to PIL image
        enhanced_img = Image.fromarray(img_rescale)

        # Optionally, further enhance contrast using ImageEnhance
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(2)  # Increase contrast by a factor of 2

        # Save the image
        enhanced_img.save(export_path)

# Example usage
input_path = 'a/5.png' #'results/BBDM_n98_s256x256_z88_e10/BrownianBridge/sample_to_eval/200/img_TOW143__slice_10_10/output_1.png'
export_path = '5.png'
improve_mri_contrast(input_path, export_path)
