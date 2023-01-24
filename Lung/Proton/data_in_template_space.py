import ants
import glob
import os
import numpy as np

base_directory = "/Users/ntustison/Data/HeliumLungStudies2/ProtonMasks/DeepLearning/"
input_data_directory = base_directory + "Data/"
output_data_directory = base_directory + "DataInTemplateSpace/"

input_lobe_mask_files = glob.glob(input_data_directory + "*Lobes.nii.gz")

template = ants.image_read(base_directory + "protonLungTemplate.nii.gz")
center_of_mass_template = ants.get_center_of_mass(template * 0 + 1)

for i in range(len(input_lobe_mask_files)):
    
    input_image_file = input_lobe_mask_files[i].replace("Lobes", "H1")
    input_image = ants.image_read(input_image_file)

    input_lobes = ants.image_read(input_lobe_mask_files[i])    
    center_of_mass_image = ants.get_center_of_mass(input_image * 0 + 1)

    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)

    warped_image = ants.apply_ants_transform_to_image(xfrm, input_image, template)
    warped_lobes = ants.apply_ants_transform_to_image(xfrm, input_lobes, template)

    output_image_file = input_image_file.replace(input_data_directory, output_data_directory)
    output_lobes_file = input_lobe_mask_files[i].replace(input_data_directory, output_data_directory)
    print("Writing " + output_image_file)
    ants.image_write(warped_image, output_image_file)
    print("Writing " + output_lobes_file)
    ants.image_write(warped_lobes, output_lobes_file)
    
