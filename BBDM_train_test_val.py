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
folders=['TOW011', 'TOW012', 'TOW013', 'TOW015', 'TOW016', 'TOW017', 'TOW018', 'TOW019', 'TOW027', 'TOW029', 'TOW033', 'TOW037', 'TOW040',
          'TOW042', 'TOW044', 'TOW046', 'TOW048', 'TOW049', 'TOW054', 'TOW055', 'TOW057', 'TOW063', 'TOW068', 'TOW073', 'TOW074', 'TOW075', 
          'TOW080', 'TOW082', 'TOW084', 'TOW088', 'TOW092', 'TOW097', 'TOW100', 'TOW106', 'TOW112', 'TOW113', 'TOW118', 'TOW119', 'TOW125', 
          'TOW128', 'TOW130', 'TOW132', 'TOW137', 'TOW139', 'TOW140', 'TOW141', 'TOW142', 'TOW143', 'TOW144', 'TOW146', 'TOW201', 'TOW213', 
          'TOW216', 'TOW218', 'TOW224', 'TOW239', 'TOW246', 'TOW247', 'TOW248', 'TOW251', 'TOW254', 'TOW257', 'TOW259', 'TOW286', 'TOW321', 
          'TOW363', 'TOW503', 'TOW512', 'TOW513', 'TOW523', 'TOW544', 'TOW549', 'TOW553', 'TOW557', 'TOW563', 'TOW571', 'TOW600', 'TOW605', 
          'TOW700', 'TOW_VOL01', 'TOW_VOL02', 'TOW_VOL03', 'TOW_VOL04', 'TOW_VOL05', 'TOW_VOL06', 'TOW_VOL07', 'TOW_VOL08', 'TOW_VOL09', 
          'TOW_VOL10', 'TOW_VOL11', 'TOW_VOL12', 'TOW_VOL13', 'TOW_VOL14', 'TOW_VOL15', 'TOW_VOL16', 'TOW_VOL18', 'TOW_VOL19', 'TOW_VOL20']
patients = folders[0:79]
volunteers = folders[79:99]
trainfolders = patients[0:47]+volunteers[0:11]
testfolders = patients[47:63]+volunteers[11:15]
valfolders = patients[63:80]+volunteers[15:19]
print("train",trainfolders)
# ['TOW011', 'TOW012', 'TOW013', 'TOW015', 'TOW016', 'TOW017', 'TOW018', 'TOW019', 'TOW027', 'TOW029', 'TOW033', 'TOW037', 
# 'TOW040', 'TOW042', 'TOW044', 'TOW046', 'TOW048', 'TOW049', 'TOW054', 'TOW055', 'TOW057', 'TOW063', 'TOW068', 'TOW073', 
# 'TOW074', 'TOW075', 'TOW080', 'TOW082', 'TOW084', 'TOW088', 'TOW092', 'TOW097', 'TOW100', 'TOW106', 'TOW112', 'TOW113', 
# 'TOW118', 'TOW119', 'TOW125', 'TOW128', 'TOW130', 'TOW132', 'TOW137', 'TOW139', 'TOW140', 'TOW141', 'TOW142', 'TOW_VOL01', 
# 'TOW_VOL02', 'TOW_VOL03', 'TOW_VOL04', 'TOW_VOL05', 'TOW_VOL06', 'TOW_VOL07', 'TOW_VOL08', 'TOW_VOL09', 'TOW_VOL10', 'TOW_VOL11']
print("test",testfolders)
# ['TOW143', 'TOW144', 'TOW146', 'TOW201', 'TOW213', 'TOW216', 'TOW218', 'TOW224', 'TOW239', 'TOW246', 'TOW247', 'TOW248', 
# 'TOW251', 'TOW254', 'TOW257', 'TOW259', 'TOW_VOL12', 'TOW_VOL13', 'TOW_VOL14', 'TOW_VOL15']
print("val",valfolders)
# ['TOW286', 'TOW321', 'TOW363', 'TOW503', 'TOW512', 'TOW513', 'TOW523', 'TOW544', 'TOW549', 'TOW553', 'TOW557', 'TOW563', 
# 'TOW571', 'TOW600', 'TOW605', 'TOW700', 'TOW_VOL16', 'TOW_VOL18', 'TOW_VOL19', 'TOW_VOL20']

### CHANGE THIS TO DESIRED FOLDER
# Define the source folders and the destination folders
source_folders = {'velx': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/4dflow_slices_n98_s320x320_z88_velx', 
                  'vely': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/4dflow_slices_n98_s320x320_z88_vely', 
                  'velz': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/4dflow_slices_n98_s320x320_z88_velz'}
destination_folders = {'velx': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s320x320_z88_velx/train',
                       'vely': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s320x320_z88_vely/train',
                       'velz': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s320x320_z88_velz/train'}

# Function to copy files for the first 20 scans
def copy_files_for_scans(source, destination):
    # Loop through the first 2s0 scans
    for scan in trainfolders: ### CHANGE THIS
        # For each scan, loop through the slices
        for N in range(1, 89):
            for M in range(1, 16):
                file_name = f'img_{scan}__slice_{N}_{M}.jpg'
                source_file = os.path.join(source, file_name)
                destination_file = os.path.join(destination, file_name)
                # Copy file if it exists
                if os.path.exists(source_file):
                    shutil.copy(source_file, destination_file)
                else:
                    print(f"File does not exist: {source_file}")

# Run the copy function for each folder
for key in source_folders:
    copy_files_for_scans(source_folders[key], destination_folders[key])

print("Finished")