finalfolders =  ['TOW700', 'TOW_VOL01']
path = "my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/4dflow"
outputdir = 'my-rdisk/r-divi/RNG/Projects/stages/Pim/Doris/Data/processed/4dflow_cropped'
target_shape = (320, 320, 88, 15)  # Desired shape after padding
pattern = re.compile(r'TOW(?:_VOL)?(\d+)_')

for folder in finalfolders:
    for file in os.listdir(path+"/"+folder):
        file4d = file
        img = nib.load(path+"/"+folder+"/"+file4d).get_fdata()
        