import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# Paths
input_base_dir = '/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/raw/segmentations'
output_base_dir = '/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/seg_slices_n98_s320x320_z88'

# Create the output directory if it does not exist
os.makedirs(output_base_dir, exist_ok=True)

# Function to extract the index after 'card' which is not zero
def extract_card_index(file_name):
    match = re.search(r'card0*([1-9]\d*)', file_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot find a valid card index in file name: {file_name}")

# Loop through all folders in the input directory
for folder_name in tqdm(os.listdir(input_base_dir), desc='Processing folders'):
    folder_path = os.path.join(input_base_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue  # Skip if not a directory

    # Loop through all .nii.gz files in the current folder
    for file_name in tqdm(sorted(os.listdir(folder_path)), desc=f'Processing files in {folder_name}', leave=False):
        if not file_name.endswith('.nii.gz'):
            continue  # Skip non .nii.gz files
        
        # Extract the file index from the file name using the custom function
        file_index = extract_card_index(file_name)
        
        # Load the .nii.gz file
        file_path = os.path.join(folder_path, file_name)
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        
        # Ensure data is at least 3D
        if data.ndim != 3:
            raise ValueError(f"Unexpected data shape {data.shape} in file {file_path}")

        # Create the full set of slices, padded to 320x320
        slices = []

        for i in range(data.shape[2]):
            slice_2d = data[:, :, i]

            # Calculate padding sizes to center the slice in a 320x320 array
            pad_height = (320 - slice_2d.shape[0]) // 2
            pad_width = (320 - slice_2d.shape[1]) // 2
            padded_slice = np.pad(slice_2d,
                                  ((pad_height, 320 - slice_2d.shape[0] - pad_height),
                                   (pad_width, 320 - slice_2d.shape[1] - pad_width)),
                                  mode='constant')
            slices.append(padded_slice)

        # Add zero-padded images if necessary to make up 88 slices
        num_slices = len(slices)
        if num_slices < 88:
            num_padding = 88 - num_slices
            padding_slices = [np.zeros((320, 320))] * num_padding
            half_padding = num_padding // 2
            slices = padding_slices[:half_padding] + slices + padding_slices[half_padding:]

        # Loop through the slices and save each one as a PNG file
        for i, slice_2d in enumerate(slices):
            output_file_name = f"img_{folder_name}_slice_{i + 1}_{file_index}.png"
            output_file_path = os.path.join(output_base_dir, output_file_name)

            plt.imsave(output_file_path, slice_2d, cmap='gray', format='png')

    print(f"All files in folder '{folder_name}' have been processed.")

print("Processing complete.")
