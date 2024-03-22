import ants
import antspynet
import numpy as np

import os
import glob

from batch_generator import batch_generator

base_directory = '/home/ntustison/Data/EXACT09/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'Nifti/training/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

ct_files = glob.glob(data_directory + "images/*.nii.gz")

training_ct_files = list()
training_seg_files = list()
training_label_files = list()

for i in range(len(ct_files)):

    ct_file = ct_files[i] 
    seg_file = ct_file.replace("images", "masks")
    label_file = ct_file.replace("images", "annotations")

    if(os.path.exists(ct_file) and os.path.exists(seg_file)) and os.path.exists(label_file):
        training_ct_files.append(ct_file)
        training_seg_files.append(seg_file)
        training_label_files.append(label_file)

print("Total training image files: ", len(training_ct_files))

print( "Training")

batch_size = 4
patch_size = (160, 160, 160)
channel_size = 1  

generator = batch_generator(batch_size=batch_size,
                    ct_files=training_ct_files,
                    seg_files=training_seg_files,
                    label_files=training_label_files,
                    patch_size=patch_size,
                    number_of_channels=channel_size,
                    do_histogram_intensity_warping=False,
                    do_simulate_bias_field=False,
                    do_add_noise=False,
                    do_random_transformation=False)


X, Y, W = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX_ct_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,1])), "batchX_left_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,2])), "batchX_right_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,3])), "batchX_air_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:,1])), "batchY_" + str(i) + ".nii.gz")    

print(X.shape)
print(len(Y))


