import ants
import antspynet
import numpy as np

import os
import glob

from batch_generator import batch_generator

base_directory = '/home/ntustison/Data/BRATS/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'TCIA/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_files = glob.glob(data_directory + "images_structural_unstripped/*/*_T1_*.nii.gz")

training_t1_files = list()
training_t2_files = list()
training_flair_files = list()
training_t1gd_files = list()
training_brain_mask_files = list()
training_seg_files = list()

for i in range(len(t1_files)):

    t1_file = t1_files[i] 
    t2_file = t1_file.replace("T1", "T2")
    t1gd_file = t1_file.replace("T1", "T1GD")
    flair_file = t1_file.replace("T1", "FLAIR") 

    mask_file = t1_file.replace("images_structural_unstripped", "ants_brain_extraction")
    mask_file = mask_file.replace("T1_unstripped", "T1_brain_mask")

    seg_file = t1_file.replace("images_structural_unstripped", "automated_segm")
    seg_file = seg_file.replace("T1_unstripped", "automated_approx_segm")
    seg_file = os.path.normpath(os.path.dirname(seg_file) + "/../") + "/" + os.path.basename(seg_file)

    if(os.path.exists(t1_file) and 
       os.path.exists(t2_file) and
       os.path.exists(t1gd_file) and
       os.path.exists(flair_file) and
       os.path.exists(mask_file) and
       os.path.exists(seg_file)):

        training_t1_files.append(t1_file)
        training_t2_files.append(t2_file)
        training_t1gd_files.append(t1gd_file)
        training_flair_files.append(flair_file)
        training_brain_mask_files.append(mask_file)
        training_seg_files.append(seg_file)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 5

generator = batch_generator(batch_size=batch_size,
                    flair_files=training_flair_files,
                    t1_files=training_t1_files,
                    t1gd_files=training_t1gd_files,              
                    t2_files=training_t2_files,
                    brain_mask_files=training_brain_mask_files,
                    seg_files=training_seg_files,
                    patch_size=(64, 64, 64),
                    number_of_channels=5,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=False,
                    do_add_noise=False,
                    do_random_transformation=False)


X, Y, W = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX_t2_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,1])), "batchX_t1_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,4])), "batchX_M_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchX_y_" + str(i) + ".nii.gz")    
    for j in range(Y.shape[-1]):
        ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:,j])), "batchX_y_" + str(i) + "_" + str(j) + ".nii.gz")

print(X.shape)
print(len(Y))


