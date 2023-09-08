import ants
import antspynet
import numpy as np

import os
import glob

from batch_generator import batch_generator

base_directory = '/Users/ntustison/Data/Public/UKBiobank/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "WmhTrainingData/"

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "*/T1.nii.gz")

training_t1_files = list()
training_t2_files = list()
training_atropos_files = list()
training_sysu_files = list()
training_bianca_files = list()

for i in range(len(t1_images)):

    subject_directory = os.path.dirname(t1_images[i])

    training_t1_files.append(t1_images[i])
    training_t2_files.append(t1_images[i].replace("T1", "T2_FLAIR"))
    training_atropos_files.append(t1_images[i].replace("T1", "atropos"))
    training_sysu_files.append(t1_images[i].replace("T1", "sysu"))
    training_bianca_files.append(t1_images[i].replace("T1", "bianca"))

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 5

generator = batch_generator(batch_size=batch_size,
                    t2_files=training_t2_files,
                    t1_files=training_t1_files,
                    atropos_files=training_atropos_files,
                    sysu_files=training_sysu_files,
                    bianca_files=training_bianca_files,
                    patch_size=(64, 64, 64),
                    do_histogram_intensity_warping=False,
                    do_simulate_bias_field=False,
                    do_add_noise=False,
                    do_random_transformation=False)


X, Y, W = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX_t2_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,1])), "batchX_t1_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,2])), "batchX_wm_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:,0])), "batchX_y_" + str(i) + ".nii.gz")

print(X.shape)
print(len(Y))


