import nibabel as nib
import matplotlib.pylab as plt
import os

# print(os.getcwd())
# print(os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis"))
# files4dflow = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/4dflow")
# files3dcine = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine")
# print("4dflow",files4dflow)
# print("3dcine",files3dcine)

# for element in files4dflow:
#     if element not in files3dcine:
#         print("not in", element)


# for element in files3dcine:
#     if element not in files4dflow:
#         print(element)        


# in cine but not in flow: TOW036, Preparations, TOW010, TOW108, TOW003, TOW001        
# print(len(files3dcine))
# print(len(files4dflow)) 
# use all the files in 4d flow, 104 datapoints for now. check which folders and which shapes
# print(len(os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine"))) 
# count = 0
# for folder in os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine"):
#     if len(os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine/"+folder)) ==1:
#         count+=1
#         print(folder)

# # print(count) # 95 with only one file, then that must be the right file
# subjects = []
# path = "/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine"
# sameshapecounter=0
# for folder in os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/4dflow"):
#     if len(os.listdir(path+"/"+folder)) ==1:
#         filename = os.listdir(path+"/"+folder)[0]
#         img = nib.load(path+"/"+folder+"/"+filename).get_fdata()
#         print(img.shape)
#         if img.shape == (256, 256, 70, 15):
#             subjects.append(folder)

#             # sameshapecounter+=1
# print(subjects)
# # print(sameshapecounter)
# folders = ['TOW037', 'TOW106', 'TOW063', 'TOW068', 'TOW033', 'TOW_VOL03', 'TOW_VOL01', 'TOW224', 'TOW363', 'TOW_VOL20', 'TOW248', 'TOW218', 'TOW048', 
#             'TOW216', 'TOW_VOL16', 'TOW_VOL04', 'TOW013', 'TOW254', 'TOW_VOL13', 'TOW_VOL07', 'TOW_VOL09', 'TOW132', 'TOW125', 'TOW049', 'TOW_VOL10', 
#             'TOW259', 'TOW080', 'TOW257', 'TOW042', 'TOW_VOL002', 'TOW113', 'TOW097', 'TOW040', 'TOW_VOL15', 'TOW513', 'TOW251', 'TOW046', 'TOW_VOL05', 
#             'TOW201', 'TOW_VOL18', 'TOW_VOL06', 'TOW_VOL12', 'TOW012', 'TOW017', 'TOW011', 'TOW018']
# finalfolders=[]
# path = "/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/4dflow"
# sameshapecounter=0
# for folder in folders:
#     for file in os.listdir(path+"/"+folder):
#         if "rec-mag" in file:
#             file4d = file
#             # print(file4d)
#             img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
#             # print(img.shape)
#             if img.shape == (256, 256, 70, 30):
#                 # print(img.shape)
#                 # sameshapecounter+=1\
#                 finalfolders.append(folder)

# print(finalfolders)

# only 34 with the same shape, but as we will be processing in 4D will have more data
# then 2380 image slices in the folder 4dflow


###### OLD ######
# finalfolders = ['TOW037', 'TOW106', 'TOW068', 'TOW033', 'TOW_VOL03', 'TOW_VOL01', 'TOW224', 'TOW048', 'TOW216', 'TOW_VOL04', 'TOW013', 'TOW254', 'TOW_VOL07',
#                  'TOW_VOL09', 'TOW132', 'TOW125', 'TOW049', 'TOW_VOL10', 'TOW259', 'TOW080', 'TOW257', 'TOW042', 'TOW_VOL002', 'TOW113', 'TOW097', 'TOW040', 
#                  'TOW251', 'TOW046', 'TOW_VOL05', 'TOW201', 'TOW_VOL06', 'TOW017', 'TOW011', 'TOW018']
# deze precies zo processen maar storen in processed map als jpeg. Dan kunnen trainen :)
# dit worden dus alle 34 met 256x256 shape, proberen zo te runnen en alle 64 in configs file naar 256 doen en proberen, anders reshapen
from PIL import Image
import re
import numpy as np



# ###### 4DFLOW ######
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

# Example usage
# count 98 is correct and verified
# processing is verified, output will be quite some black slices, think of impact of this
target_shape = (320, 320, 88, 15)  # Desired shape after padding

# finalfolders = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow")
# path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow"
# outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow'
# pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

# for folder in finalfolders:
#     for file in os.listdir(path+"/"+folder):
#         if "-mag" in file:
#             file4d = file
#             img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
#             img = img[:,:,:,::2]
#             img_arr = np.array(img)
#             img_arr = pad_to_size(img_arr, target_shape)
#             # print(img_arr[160,160,40,:])
#             norm_img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
#             norm_img_arr = norm_img_arr.astype(np.uint8)
#             img = norm_img_arr
#             # print(img[160,160,40,:])
#             match = pattern.search(file4d)
#             for i in range(img.shape[2]):
#                 for j in range(img.shape[3]):
#                     slice_2d = img[:,:,i,j]
#                     image = Image.fromarray(slice_2d)
#                     # print(image.mode)
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
#                     filename = f'img_{match.group(0)}_slice_{i+1}_{j+1}.jpg'
#                     file_path = os.path.join(outputdir, filename)
#                     print(file4d, slice_2d.shape, i, j, filename)
#                     image.save(file_path, 'JPEG')
#     print(folder)
# print('Finished')

# finalfolders = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow")
# path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/3d_cine"
# outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/bSSFP'
# pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

# for folder in finalfolders:
#     for file in os.listdir(path+"/"+folder):
#         file3d = file
#         img = nib.load(path+"/"+folder+"/"+file3d).get_fdata()
#         img_arr = np.array(img)
#         img_arr = pad_to_size(img_arr, target_shape)
#         norm_img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
#         norm_img_arr = norm_img_arr.astype(np.uint8)
#         img = norm_img_arr
#         match = pattern.search(file3d)
#         for i in range(img.shape[2]):
#             for j in range(img.shape[3]):
#                 slice_2d = img[:,:,i,j]
#                 image = Image.fromarray(slice_2d)
#                 if image.mode != 'RGB':
#                     image = image.convert('RGB')
#                 filename = f'img_{match.group(0)}_slice_{i+1}_{j+1}.jpg'
#                 file_path = os.path.join(outputdir, filename)
#                 # print(file3d, slice_2d.shape, i, j, filename)
#                 image.save(file_path, 'JPEG')
#     print(folder)
# print('Finished')



# ###### bSSFP ######
# print("starting")
# path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/3d_cine"
# outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/bSSFP'

# finalfolders = ['TOW037', 'TOW106', 'TOW068', 'TOW033', 'TOW_VOL03', 'TOW_VOL01', 'TOW224', 'TOW048', 'TOW216', 'TOW_VOL04', 'TOW013', 'TOW254', 'TOW_VOL07',
#                  'TOW_VOL09', 'TOW132', 'TOW125', 'TOW049', 'TOW_VOL10', 'TOW259', 'TOW080', 'TOW257', 'TOW042', 'TOW_VOL002', 'TOW113', 'TOW097', 'TOW040', 
#                  'TOW251', 'TOW046', 'TOW_VOL05', 'TOW201', 'TOW_VOL06', 'TOW017', 'TOW011', 'TOW018']
# pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

# for folder in finalfolders:
#     for file in os.listdir(path+"/"+folder):
#         file4d = file
#         img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
#         # img = img[:,:,:,::2]
#         img_arr = np.array(img)
#         norm_img_arr = 255 * (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
#         norm_img_arr = norm_img_arr.astype(np.uint8)
#         img = norm_img_arr
#         # img = img[:,:,:,0]
# #           print(file4d)
# #           print(img.shape)
# #           print(type(resized_img_2))
#         match = pattern.search(file4d)
#         for i in range(img.shape[2]):
#             for j in range(img.shape[3]):
#                 slice_2d = img[:,:,i,j]
#                 image = Image.fromarray(slice_2d)
#                 # print(image.mode)

#                 if image.mode != 'RGB':
#                     image = image.convert('RGB')
#                 filename = f'img_{match.group(0)}_slice_{i+1}_{j+1}.jpg'
#                 file_path = os.path.join(outputdir, filename)
#                 print(file4d, slice_2d.shape, i, j, filename)
#                 image.save(file_path, 'JPEG')


# print("Finished")


###### Folders in right location ######

folders=['TOW011', 'TOW012', 'TOW013', 'TOW015', 'TOW016', 'TOW017', 'TOW018', 'TOW019', 'TOW027', 'TOW029', 'TOW033', 'TOW037', 'TOW040',
          'TOW042', 'TOW044', 'TOW046', 'TOW048', 'TOW049', 'TOW054', 'TOW055', 'TOW057', 'TOW063', 'TOW068', 'TOW073', 'TOW074', 'TOW075', 
          'TOW080', 'TOW082', 'TOW084', 'TOW088', 'TOW092', 'TOW097', 'TOW100', 'TOW106', 'TOW112', 'TOW113', 'TOW118', 'TOW119', 'TOW125', 
          'TOW128', 'TOW130', 'TOW132', 'TOW137', 'TOW139', 'TOW140', 'TOW141', 'TOW142', 'TOW143', 'TOW144', 'TOW146', 'TOW201', 'TOW213', 
          'TOW216', 'TOW218', 'TOW224', 'TOW239', 'TOW246', 'TOW247', 'TOW248', 'TOW251', 'TOW254', 'TOW257', 'TOW259', 'TOW286', 'TOW321', 
          'TOW363', 'TOW503', 'TOW512', 'TOW513', 'TOW523', 'TOW544', 'TOW549', 'TOW553', 'TOW557', 'TOW563', 'TOW571', 'TOW600', 'TOW605', 
          'TOW700', 'TOW_VOL01', 'TOW_VOL02', 'TOW_VOL03', 'TOW_VOL04', 'TOW_VOL05', 'TOW_VOL06', 'TOW_VOL07', 'TOW_VOL08', 'TOW_VOL09', 
          'TOW_VOL10', 'TOW_VOL11', 'TOW_VOL12', 'TOW_VOL13', 'TOW_VOL14', 'TOW_VOL15', 'TOW_VOL16', 'TOW_VOL18', 'TOW_VOL19', 'TOW_VOL20']
# split
# train: 79, 47, 16, 16
# test/val: 19, 11, 4 ,4

patients = folders[0:79]
volunteers = folders[79:99]
trainfolders = patients[0:47]+volunteers[0:11]
testfolders = patients[47:63]+volunteers[11:15]
valfolders = patients[63:80]+volunteers[15:19]
# print("train",train)
# print("test",test)
# print("val",val)

# ## Adapt folder name and path and use this to store data in the right folders
### SET VAR
import os
import shutil

# Define the source folders and the destination folders
source_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow', 'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/bSSFP'}
destination_folders = {'4dflow': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/processed2/val/A',
                       'bSSFP': 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/BBDM_Thesis/data/processed2/val/B'}

# Function to copy files for the first 20 scans
def copy_files_for_scans(source, destination):
    # Loop through the first 20 scans
    for scan in valfolders: 
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

