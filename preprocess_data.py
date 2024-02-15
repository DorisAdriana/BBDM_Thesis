from PIL import Image
import os

# Path to the directory containing the images
source_list =['data/Cdata/train/A', 'data/Cdata/train/B', 'data/Cdata/test/A', 'data/Cdata/test/B', 'data/Cdata/val/A', 'data/Cdata/val/B']
source_list =['data/Cdata/train/A'] #, 'data/Cdata/train/B', 'data/Cdata/test/A', 'data/Cdata/test/B', 'data/Cdata/val/A', 'data/Cdata/val/B']

# Target size
target_size = (64, 64)
filecount=0

# Iterate over all files in the directory
for source_dir in source_list:
    for filename in os.listdir(source_dir):
        filecount += 1
        #if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):  # Check for JPEG files
            # Construct the full file path
        file_path = os.path.join(source_dir, filename)
            
            # Open the image
        with Image.open(file_path) as img:
                # Resize the image
            resized_img = img.resize(target_size)
                
                # Save the resized image back to the same location (or you can choose a different directory)
            resized_img.save(file_path)

print("All JPEG images have been resized to 64x64 pixels.")
print(filecount)
print(os.listdir(source_dir))

# for source_dir in source_list:
#     for filename in os.listdir(source_dir):
#         file_path = os.path.join(source_dir, filename)
#         # Open the image
#         with Image.open(file_path) as img:
#             print(img.size)
                