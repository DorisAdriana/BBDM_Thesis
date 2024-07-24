import os
from PIL import Image
import numpy as np
import nibabel as nib

def create_nifti_from_slices(num_scans, num_slices_a, num_slices_b, scan_ids, output_directory, input_directory, data_type='pred'):
    """
    Creates NIfTI images from slices stored as .png files, organizing them according to scan IDs and slice types.
    
    Parameters:
    - num_scans: Number of scans.
    - num_slices_a: Number of slice types 'a'.
    - num_slices_b: Number of slice types 'b'.
    - scan_ids: List of scan identifiers.
    - output_directory: Directory where the output NIfTI files will be saved.
    - input_directory: Directory where the input .png files are located.
    - data_type: Specifies the type of data ('gt' for ground truth, 'pred' for prediction).
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    for scan_id in scan_ids:
        for b in range(1, num_slices_b + 1):
            slice_a_images = []
            for a in range(1, num_slices_a + 1):
                if data_type == 'pred':
                    file_name = f"img_TOW{scan_id}__slice_{a}_{b}.jpg"
                    file_path = os.path.join(input_directory, file_name)
                    img = Image.open(file_path)
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    #print(img_array.shape)
                    slice_a_images.append(img_array)
                    found_image = True
                    break
            
            if slice_a_images:
                # For each 'b', create a 3D array (stacking along the third dimension, axis=2)
                slice_b_array = np.stack(slice_a_images, axis=2)
                
                # Convert the 3D array into a NIfTI image
                nifti_img = nib.Nifti1Image(slice_b_array, affine=np.eye(4))
                data = nifti_img.get_fdata()
                # print(data.shape)
                
                # Save the NIfTI image to a .nii.gz file, named according to the scan and 'b' value
                output_path = os.path.join(output_directory, f"scan_{scan_id}_b{b:02}.nii.gz")
                nib.save(nifti_img, output_path)
                print(f"Saved {output_path}.")
            else:
                print(f"Warning: No images were found for 'b'={b} in scan {scan_id}. Skipping this 'b' value.")

create_nifti_from_slices(
    num_scans=20, 
    num_slices_a=88, 
    num_slices_b=15, 
    scan_ids= ['_VOL12', '_VOL13', '_VOL14', '_VOL15', '143', '144','146', '201','213', '216', '218', '224','239', '246', '247', '248', '251', '254', '257', '259'],# ['143', '144', '146', '201', '213', '216','218', '224'],
    output_directory="my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/segmentations/inputs/Unet/pred_niftis", 
    input_directory= 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/Unetbaseline_dropout/results/predictions/10_epoch', #/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/results/BBDM_n98_s256x256_z88_e10/BrownianBridge/sample_to_eval/200/',#"/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/bSSFP_slices_n98_s320x320_z88", #'/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/results/BBDM_60e_98s_256/BrownianBridge/sample_to_eval/200/', # for gt: "/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/bSSFP_slices_n98_s320x320_z88",
    data_type= 'pred'  # 'gt' or 'pred'
)