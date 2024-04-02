import nibabel as nib
from PIL import Image
import numpy as np
finalfolders =  ['TOW700', 'TOW_VOL01']
path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow"
outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow_cropped'
target_shape = (320, 320, 88, 15)  # Desired shape after padding
# pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

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
