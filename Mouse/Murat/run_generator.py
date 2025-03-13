
import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

import numpy as np

from batch_generator import batch_generator

base_directory = '/home/ntustison/Data/Mouse/MuratMaga/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template_file = base_directory + "image.nii.gz"
labels_file = base_directory + "labels_reduced.nii.gz"

template = ants.image_read(template_file)
labels = ants.image_read(labels_file)

new_spacing = (0.05, 0.05, 0.05)
new_shape = (256, 256, 256)

template = ants.resample_image(template, new_spacing, use_voxels=False, interp_type=4)
labels = ants.resample_image(labels, new_spacing, use_voxels=False, interp_type=1)

template = antspynet.pad_or_crop_image_to_size(template, new_shape)
labels = antspynet.pad_or_crop_image_to_size(labels, new_shape)

unique_labels = np.unique(labels.numpy())

template = ants.iMath_normalize(template)

#
# Set up the training generator
#

batch_size = 2

ants.image_write(template, "template.nii.gz")
ants.image_write(labels, "labels.nii.gz")

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    images=[template],
                    labels=[labels],
                    unique_labels=unique_labels,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction=None,
                    run_generator=True)

(X, Y) = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchY_" + str(i) + ".nii.gz")

print(X.shape)
print(len(Y))



