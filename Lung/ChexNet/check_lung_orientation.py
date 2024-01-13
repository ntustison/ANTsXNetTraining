import ants
import antspynet
import numpy as np
import glob
import os

base_directory = '/Users/ntustison/Data/Public/Chexnet/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Nifti/"

image_files = glob.glob(data_directory + "/images_*/*.nii.gz")

count = 0
for i in range(len(image_files)):
    print("Processing ", i, " out of ", len(image_files))
    image_file = image_files[i]
    
    image = ants.image_read(image_file)
    reoriented_image = antspynet.check_xray_lung_orientation(image, verbose=True)
    
    if reoriented_image is not None:
        output_directory = (os.path.dirname(image_file)).replace("Nifti", "NiftiReoriented")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        output_file = output_directory + "/" + os.path.basename(image_file)    
        print(image_file, " needs to be reoriented.")
        print("Writing ", output_file)
        ants.image_write(reoriented_image, output_file)
        count = count + 1

print("Count = ", count)        
        