import ants
import numpy as np
import antspynet

from batch_generator import batch_generator

import glob
import os
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

base_directory = '/home/ntustison/Data/BrainExtraction/'
scripts_directory = base_directory + 'Scripts/'

template = ants.image_read(antspynet.get_antsxnet_data("nki"))


print("Loading braindata.")

# data_directory = base_directory + '/BrainWeb/NiftiImages/'

# training_image_files = glob.glob(data_directory + "*t1w_p4.nii.gz")
# training_segmentation_files = list()

# if len(training_image_files) == 0:
#     print(data_directory)
#     raise ValueError("No training images.")

# for i in range(len(training_image_files)):
#     segmentation_file = training_image_files[i].replace("t1w_p4", "crisp_v_consolidated")
#     training_segmentation_files.append(segmentation_file)

training_image_files = list()
training_segmentation_files = list()
training_image_files.append(base_directory + "BrainWeb/Template/T_template0.nii.gz")
training_segmentation_files.append(base_directory + "BrainWeb/Template/T_template0Mask_Phil.nii.gz")
training_image_files.append(base_directory + "BrainWeb/Template_Reoriented/T_template0.nii.gz")
training_segmentation_files.append(base_directory + "BrainWeb/Template_Reoriented/T_template0Mask_Phil.nii.gz")

print("Total training image files: ", len(training_image_files))
print( "Training")


###
#
# Set up the training generator
#
batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    training_image_files=training_image_files,
                    training_segmentation_files=training_segmentation_files,
                    do_histogram_equalization=True,
                    do_histogram_rank_intensity=True,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    verbose=True)

X, Y = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    for j in range(X.shape[-1]):
        ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,j])), "batchX" + str(i) + "_" + str(j) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchY_" + str(i) + ".nii.gz")

