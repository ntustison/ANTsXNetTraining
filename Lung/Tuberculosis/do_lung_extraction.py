import ants
import antspynet
import numpy as np
import glob
import os

base_directory = '/Users/ntustison/Data/Public/Chexnet/Tuberculosis/OrganizedForANTsXNetTraining/'
data_directory = base_directory + "Images/"

image_files = glob.glob(data_directory + "TB_Chest_Radiography_Database/*/*.nii.gz")

for i in range(len(image_files)):
    print("Processing ", i, " out of ", len(image_files))
    image_file = image_files[i]
    
    mask_file = image_files[i].replace("Images", "Masks")
    if os.path.exists(mask_file):
        continue
    else:
        subject_directory = os.path.dirname(mask_file)
        if not os.path.exists(subject_directory):
            os.makedirs(subject_directory, exist_ok=True)
        image = ants.image_read(image_file)
        extraction = antspynet.lung_extraction(image, modality="xray", verbose=True)    
        ants.image_write(extraction['segmentation_image'], mask_file)
        