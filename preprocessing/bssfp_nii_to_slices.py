import nibabel as nib
import matplotlib.pylab as plt
import os
from PIL import Image
import re
import numpy as np


### Notes
# 98 subjects

### Functions
# padding function
def pad_to_size(arr, target_shape):
    """
    Pads a 4D array to the specified target shape with zeros.
    
    Parameters:
    - arr: numpy.ndarray, the input 4D array to pad.
    - target_shape: tuple of int, the target shape to pad the array to. Should be of the form (n, c, h, w) where
      n is the target size for the first dimension,
      c is the target size for the second dimension,
      h is the target size for the third dimension, and
      w is the target size for the fourth dimension.
      
    Returns:
    - Padded numpy.ndarray of shape target_shape.
    """
    # Ensure the array is 4D and target_shape is valid
    assert arr.ndim == 4, "Input array must be 4-dimensional."
    assert len(target_shape) == 4, "Target shape must have 4 dimensions."
    
    # Calculate the padding needed for each dimension
    pad_width = []
    for current_size, target_size in zip(arr.shape, target_shape):
        total_pad = max(target_size - current_size, 0)
        # Distribute padding equally on both sides
        pad_width.append((total_pad // 2, total_pad - total_pad // 2))
    
    # Apply padding
    padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded_arr

###### bSSFP ######
finalfolders = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow")
path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/3d_cine"
outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/bSSFP_cropped'
target_shape = (320, 320, 88, 15)  # Desired shape after padding
pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

for folder in finalfolders:
    for file in os.listdir(path+"/"+folder):
        file3d = file
        img = nib.load(path+"/"+folder+"/"+file3d).get_fdata()
        img_arr = np.array(img)
        img_arr = pad_to_size(img_arr, target_shape)
        norm_img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
        norm_img_arr = norm_img_arr.astype(np.uint8)
        img = norm_img_arr
        match = pattern.search(file3d)
        for i in range(img.shape[2]):
            for j in range(img.shape[3]):
                slice_2d = img[:,:,i,j]
                image = Image.fromarray(slice_2d)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    # Crop the middle 80% of the image in the x direction
                    width, height = image.size
                    new_width = width * 0.8
                    left = (width - new_width) / 2
                    top = 0
                    right = (width + new_width) / 2
                    bottom = height
                    image = image.crop((left, top, right, bottom))
                    
                filename = f'img_{match.group(0)}_slice_{i+1}_{j+1}.jpg'
                file_path = os.path.join(outputdir, filename)
                # print(file3d, slice_2d.shape, i, j, filename)
                image.save(file_path, 'JPEG')
    print(folder)
print('Finished')
