import os
from PIL import Image
import numpy as np
import nibabel as nib

# Parameters
num_scans = 2  # Number of scans
num_slices_a = 88  # Number of slice types 'a'
num_slices_b = 15  # Number of slice types 'b'
scan_ids = ['143', '144']  # Example scan identifiers
output_directory = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/Unet_input" 
input_directory = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/results/m19_BBDM_60e_98s_256/BrownianBridge/sample_to_eval/200"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

for scan_id in scan_ids:
    for b in range(1, num_slices_b + 1):
        slice_a_images = []
        for a in range(1, num_slices_a + 1):
            folder_name = f"img_TOW{scan_id}__slice_{a}_{b}"
            folder_path = os.path.join(input_directory, folder_name)
            found_image = False
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".png"):
                    img_path = os.path.join(folder_path, file)
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    slice_a_images.append(img_array)
                    found_image = True
                    break
            if not found_image:
                print(f"No suitable image found in {folder_path}.")
        
        if slice_a_images:
            # For each 'b', create a 3D array (stacking along the third dimension, axis=2)
            slice_b_array = np.stack(slice_a_images, axis=2)
            
            # Convert the 3D array into a NIfTI image
            nifti_img = nib.Nifti1Image(slice_b_array, affine=np.eye(4))
            
            # Save the NIfTI image to a .nii.gz file, named according to the scan and 'b' value
            output_path = os.path.join(output_directory, f"scan_{scan_id}_b{b:02}.nii.gz")
            nib.save(nifti_img, output_path)
            print(f"Saved {output_path}.")
        else:
            print(f"Warning: No images were found for 'b'={b} in scan {scan_id}. Skipping this 'b' value.")

# export nnUNet_raw_data_base=“/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/Unet/nnUnet_raw_data”
# export nnUNet_raw_data_base=“/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/Unet/nnUnet_raw_data”
# export RESULTS_FOLDER=“/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/Unet/modelweights”

# export RESULTS_FOLDER=“/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/Unet/modelweights”