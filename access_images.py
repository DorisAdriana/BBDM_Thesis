import nibabel as nib
import matplotlib.pylab as plt
import os

# print(os.getcwd())
# print(os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg"))
# files4dflow = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/4dflow")
# files3dcine = os.listdir("/home/rnga/dawezenberg/my-rdisk/r-divi/RNG/Temp/dawezenberg/Data/3d_cine")
# print("4dflow",files4dflow)
# print("3dcine",files3dcine)

# for element in files4dflow:
#     if element not in files3dcine:
#         print(element)


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

### SET VAR
finalfolders = ['TOW037', 'TOW106', 'TOW068', 'TOW033', 'TOW_VOL03', 'TOW_VOL01', 'TOW224', 'TOW048', 'TOW216', 'TOW_VOL04', 'TOW013', 'TOW254', 'TOW_VOL07',
                 'TOW_VOL09', 'TOW132', 'TOW125', 'TOW049', 'TOW_VOL10', 'TOW259', 'TOW080', 'TOW257', 'TOW042', 'TOW_VOL002', 'TOW113', 'TOW097', 'TOW040', 
                 'TOW251', 'TOW046', 'TOW_VOL05', 'TOW201', 'TOW_VOL06', 'TOW017', 'TOW011', 'TOW018']
path = "my-rdisk/r-divi/rng/projects/stages/pim/doris/data/4dflow"


# deze precies zo processen maar storen in processed map als jpeg. Dan kunnen trainen :)
# dit worden dus alle 34 met 256x256 shape, proberen zo te runnen en alle 64 in configs file naar 256 doen en proberen, anders reshapen
from PIL import Image
outputdir = 'data/processed/B'
### CREATE 4d
for folder in finalfolders:
    for file in os.listdir(path+"/"+folder):
        if "rec-mag" in file:
            file4d = file
            # print(file4d)
            img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
            resized_img = img[:,:,:,::2]
            print(resized_img.shape)
            print(type(resized_img))
            for i in range(resized_img.shape[2]):
                for j in range(resized_img.shape[3]):
                    slice_2d = resized_img[:,:,i,j]
                    image = Image.fromarray(slice_2d)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    filename = f'slice_{i+1}_{j+1}.jpg'
                    file_path = os.path.join(outputdir, filename)
                    image.save(file_path, 'JPEG')
                    print(i)
                    print(j)
            
#

print("Finished")

