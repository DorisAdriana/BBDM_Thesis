import nibabel as nib
from PIL import Image
import numpy as np
finalfolders =  ['TOW700', 'TOW_VOL01']
path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow"
outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow_cropped'
target_shape = (320, 320, 88, 15)  # Desired shape after padding

# for folder in finalfolders:
#   for file in os.listdir(path+"/"+folder):
#        file4d = file
#       img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()

img = nib.load('/home/rnga/dawezenberg/my-scratch/outputs/scan_144_b15.nii.gz').get_fdata()
print('generated seg after unet', img.shape)
# /home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/inspectvelo.py

img = nib.load('/home/rnga/dawezenberg/my-scratch/outputs/gt/scan_144_b15.nii.gz').get_fdata()
print('gt seg after unet', img.shape)

img = nib.load('/home/rnga/dawezenberg/my-scratch/nnUNet_raw_data_base/nnUNet_raw_data/preds_Task531_3D_cine_root_branches/imagesTs/scan_144_b15_0000.nii.gz').get_fdata()
print('generated slices before unet', img.shape)

img = nib.load('/home/rnga/dawezenberg/my-scratch/nnUNet_raw_data_base/nnUNet_raw_data/Task531_3D_cine_root_branches_gt/imagesTs/scan_144_b15_0000.nii.gz').get_fdata()
print('gt before slices unet', img.shape)

#img = Image.open('/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/bSSFP_cropped/img_TOW_VOL20__slice_88_15.jpg')
#img_arr = np.array(img)
#print('generated preds', img_arr.shape)

### GT Has not been cropped yet, fix this 
# probably because the data was stored when already cropped
# so cropped data is existent just stored in a different location

img = Image.open('my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow_cropped/img_TOW_VOL12__slice_1_1.jpg')
img_arr = np.array(img)
print('ground truth input data', img_arr.shape)

img = nib.load('/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/Unet_input/scans_pred/scan_144_b15.nii.gz').get_fdata()
print('scan_pred', img.shape)
img = nib.load('/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/Unet_input/scans_gt/scan_144_b15.nii.gz').get_fdata()
print('scan_gt', img.shape)


# import nibabel as nib
# import matplotlib.pylab as plt
# import os
# from PIL import Image
# import re
# import numpy as np
# import shutil

# ###### Store data in the right format and location for training ######

# ### Subjectss
# # split
# # train: 79, 47, 16, 16
# # test/val: 19, 11, 4 ,4
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
# source_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/4dflow_slices_n98_s256x320_z88', 'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/processed/bSSFP_slices_n98_s256x320_z88'}
# destination_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s256x320_z88/train/A',
#                        'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/slices_n98_s256x320_z88/train/B'}

# # Function to copy files for the first 20 scans
# source = '/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/data/raw/bssfp'
# for folder in valfolders: ### CHANGE THIS
#     for file in os.listdir((os.path.join(source, folder))):
#         mfolder = source + "/"+folder +"/" + file
#         # print(folder)
#         img = nib.load(mfolder)
#         print(img.shape)
#         #print(folder)


