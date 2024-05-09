import nibabel as nib
import matplotlib.pylab as plt
import os
from PIL import Image
import re
import numpy as np
import shutil

###### Store data in the right format and location for training ######
### Subjectss
# split
# train: 79, 47, 16, 16
# test/val: 19, 11, 4 ,4
# folders=['TOW011', 'TOW012', 'TOW013', 'TOW015', 'TOW016', 'TOW017', 'TOW018', 'TOW019', 'TOW027', 'TOW029', 'TOW033', 'TOW037', 'TOW040',
#           'TOW042', 'TOW044', 'TOW046', 'TOW048', 'TOW049', 'TOW054', 'TOW055', 'TOW057', 'TOW063', 'TOW068', 'TOW073', 'TOW074', 'TOW075', 
#           'TOW080', 'TOW082', 'TOW084', 'TOW088', 'TOW092', 'TOW097', 'TOW100', 'TOW106', 'TOW112', 'TOW113', 'TOW118', 'TOW119', 'TOW125', 
#           'TOW128', 'TOW130', 'TOW132', 'TOW137', 'TOW139', 'TOW140', 'TOW141', 'TOW142', 'TOW143', 'TOW144', 'TOW146', 'TOW201', 'TOW213', 
#           'TOW216', 'TOW218', 'TOW224', 'TOW239', 'TOW246', 'TOW247', 'TOW248', 'TOW251', 'TOW254', 'TOW257', 'TOW259', 'TOW286', 'TOW321', 
#           'TOW363', 'TOW503', 'TOW512', 'TOW513', 'TOW523', 'TOW544', 'TOW549', 'TOW553', 'TOW557', 'TOW563', 'TOW571', 'TOW600', 'TOW605', 
#           'TOW700', 'TOW_VOL01', 'TOW_VOL02', 'TOW_VOL03', 'TOW_VOL04', 'TOW_VOL05', 'TOW_VOL06', 'TOW_VOL07', 'TOW_VOL08', 'TOW_VOL09', 
#           'TOW_VOL10', 'TOW_VOL11', 'TOW_VOL12', 'TOW_VOL13', 'TOW_VOL14', 'TOW_VOL15', 'TOW_VOL16', 'TOW_VOL18', 'TOW_VOL19', 'TOW_VOL20']
# patients = folders[0:79]
# volunteers = folders[79:99]
# trainfolders = patients[0:47]+volunteers[0:11]
# testfolders = patients[47:63]+volunteers[11:15]
# valfolders = patients[63:80]+volunteers[15:19]
# print("train",trainfolders)
# ['TOW011', 'TOW012', 'TOW013', 'TOW015', 'TOW016', 'TOW017', 'TOW018', 'TOW019', 'TOW027', 'TOW029', 'TOW033', 'TOW037', 
# 'TOW040', 'TOW042', 'TOW044', 'TOW046', 'TOW048', 'TOW049', 'TOW054', 'TOW055', 'TOW057', 'TOW063', 'TOW068', 'TOW073', 
# 'TOW074', 'TOW075', 'TOW080', 'TOW082', 'TOW084', 'TOW088', 'TOW092', 'TOW097', 'TOW100', 'TOW106', 'TOW112', 'TOW113', 
# 'TOW118', 'TOW119', 'TOW125', 'TOW128', 'TOW130', 'TOW132', 'TOW137', 'TOW139', 'TOW140', 'TOW141', 'TOW142', 'TOW_VOL01', 
# 'TOW_VOL02', 'TOW_VOL03', 'TOW_VOL04', 'TOW_VOL05', 'TOW_VOL06', 'TOW_VOL07', 'TOW_VOL08', 'TOW_VOL09', 'TOW_VOL10', 'TOW_VOL11']
# print("test",testfolders)
# ['TOW143', 'TOW144', 'TOW146', 'TOW201', 'TOW213', 'TOW216', 'TOW218', 'TOW224', 'TOW239', 'TOW246', 'TOW247', 'TOW248', 
# 'TOW251', 'TOW254', 'TOW257', 'TOW259', 'TOW_VOL12', 'TOW_VOL13', 'TOW_VOL14', 'TOW_VOL15']
# print("val",valfolders)
# ['TOW286', 'TOW321', 'TOW363', 'TOW503', 'TOW512', 'TOW513', 'TOW523', 'TOW544', 'TOW549', 'TOW553', 'TOW557', 'TOW563', 
# 'TOW571', 'TOW600', 'TOW605', 'TOW700', 'TOW_VOL16', 'TOW_VOL18', 'TOW_VOL19', 'TOW_VOL20']

# ### CHANGE THIS TO DESIRED FOLDER
# # Define the source folders and the destination folders
# source_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/4dflow_slices_n98_s256x320_z88_velx', 'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/bSSFP_slices_n98_s256x320_z88_velx'}
# destination_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s256x320_z88_velx/train/A',
#                        'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s256x320_z88_velx/train/B'}

# # Function to copy files for the first 20 scans
# source = '/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/raw/bssfp'
# for folder in valfolders: ### CHANGE THIS
#     for file in os.listdir((os.path.join(source, folder))):
#         mfolder = source + "/"+folder +"/" + file
#         # print(folder)
#         img = nib.load(mfolder)
#         print(img.shape)
#         #print(folder)

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

###### 4DFLOW ######
finalfolders = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/raw/4dflow")
path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/raw/4dflow"
outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow_slices_n98_s320x320_z88_velx'
target_shape = (320, 320, 88, 15)  # Desired shape after padding
pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

for folder in finalfolders:
    for file in os.listdir(path+"/"+folder):
        if "velx" in file:
            file4d = file
            img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
            img = img[:,:,:,::2]
            img_arr = np.array(img)
            img_arr = pad_to_size(img_arr, target_shape)
            print(img_arr.shape)
            # print(img_arr[160,160,40,:])
            norm_img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            norm_img_arr = norm_img_arr.astype(np.uint8)
            img = norm_img_arr
            # print(img[160,160,40,:])
            match = pattern.search(file4d)
            for i in range(img.shape[2]):
                for j in range(img.shape[3]):
                    slice_2d = img[:,:,i,j]
                    image = Image.fromarray(slice_2d)
                    # print(image.mode)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # # Crop the middle 80% of the image in the x direction
                    # width, height = image.size
                    # new_width = width * 0.8
                    # left = (width - new_width) / 2
                    # top = 0
                    # right = (width + new_width) / 2
                    # bottom = height
                    # image = image.crop((left, top, right, bottom))

                    filename = f'img_{match.group(0)}_slice_{i+1}_{j+1}.jpg'
                    file_path = os.path.join(outputdir, filename)
                    print(file4d, slice_2d.shape, i, j, filename)
                    image.save(file_path, 'JPEG')
    print(folder)
print('Finished')

